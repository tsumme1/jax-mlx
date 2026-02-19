"""
CNN Performance Benchmark (Julia)
Equivalent to benchmark_cnn.py using Lux, Enzyme, and Reactant on the MLX backend.

Usage:
  julia +1.11 benchmarks/benchmark_cnn.jl
"""

using Libdl, Reactant, Lux, Enzyme, Random, Statistics, Optimisers

# ── MLX Plugin Setup ──────────────────────────────────────────────────────────

function load_mlx_plugin!()
    # Find libmlx.dylib: check MLX_LIB_PATH env var, then ARGS
    mlx_lib = get(ENV, "MLX_LIB_PATH", nothing)
    if mlx_lib === nothing && !isempty(ARGS)
        mlx_lib = ARGS[1]
    end
    @assert mlx_lib !== nothing "Set MLX_LIB_PATH or pass path to libmlx.dylib as first argument"
    @assert isfile(mlx_lib) "Cannot find libmlx.dylib at $mlx_lib"
    Libdl.dlopen(mlx_lib, Libdl.RTLD_NOW | Libdl.RTLD_GLOBAL)

    # Find plugin dylib
    plugin_path = joinpath(@__DIR__, "..", "src", "jax_mlx", "libmlx_pjrt_plugin.dylib")
    if !isfile(plugin_path)
        plugin_path = joinpath(@__DIR__, "..", "build", "libmlx_pjrt_plugin.dylib")
    end
    @assert isfile(plugin_path) "Cannot find libmlx_pjrt_plugin.dylib at $plugin_path"

    # Register plugin with Reactant
    mlx_client_ptr = Reactant.XLA.PJRT.MakeClientUsingPluginAPI(plugin_path, "mlx", "MLX")
    mlx_client = Reactant.XLA.PJRT.Client(mlx_client_ptr)
    Reactant.XLA.global_backend_state.clients["mlx"] = mlx_client
    Reactant.XLA.set_default_backend(mlx_client)
    println("MLX plugin loaded ✓")
end

# Monkey-patch: the PJRT C API wrapper can't deserialize our StableHLO MLIR text
# into xla::HloModule. Since compile() discards hlo_modules anyway, a try/catch
# fallback is safe. This overrides the precompiled method via Julia's invalidation.
function Reactant.XLA.get_hlo_modules(exec::Reactant.XLA.PJRT.LoadedExecutable)
    hlo_modules = Ref{NTuple{Int64(Reactant.XLA.num_partitions(exec)),Ptr{Cvoid}}}()
    nmodules = Ref{Int32}(0)
    try
        GC.@preserve exec hlo_modules begin
            @ccall Reactant.MLIR.API.mlir_c.PjRtLoadedExecutableGetHloModules(
                exec.exec::Ptr{Cvoid}, hlo_modules::Ptr{Ptr{Cvoid}}, nmodules::Ptr{Cint}
            )::Cvoid
        end
        return map(Reactant.XLA.HloModule, hlo_modules[][1:Int(nmodules[])])
    catch e
        @warn "get_hlo_modules failed (expected for MLX plugin)" exception = e
        return Reactant.XLA.HloModule[]
    end
end

# ── CNN Model ─────────────────────────────────────────────────────────────────

function SimpleCNN()
    return Chain(
        # Block 1 (SAME padding, matching all platforms)
        # cross_correlation=true matches JAX/PyTorch semantics and avoids
        # unnecessary stablehlo.reverse ops in Enzyme's backward pass
        Conv((3, 3), 3 => 64; pad=SamePad(), cross_correlation=true),
        relu,
        Conv((3, 3), 64 => 64; pad=SamePad(), cross_correlation=true),
        relu,
        MaxPool((2, 2)),
        # Block 2
        Conv((3, 3), 64 => 128; pad=SamePad(), cross_correlation=true),
        relu,
        Conv((3, 3), 128 => 128; pad=SamePad(), cross_correlation=true),
        relu,
        MaxPool((2, 2)),
        # Block 3
        Conv((3, 3), 128 => 256; pad=SamePad(), cross_correlation=true),
        relu,
        # Global average pooling → Dense head
        GlobalMeanPool(),
        FlattenLayer(),
        Dense(256, 128, relu),
        Dense(128, 10),
    )
end

# ── Loss Function ─────────────────────────────────────────────────────────────

function cross_entropy_loss(model, ps, st, x, y)
    logits, st_ = model(x, ps, st)
    # logits is (10, batch) in Lux's format
    # log-softmax along dim 1 (classes)
    log_probs = logits .- Lux.logsumexp(logits; dims=1)
    # One-hot via broadcasting: (10,1) .== (1,batch) → (10, batch)
    classes = reshape(1:10, 10, 1)
    labels = reshape(y, 1, :)
    one_hot = Float32.(classes .== labels)
    return -sum(one_hot .* log_probs) / Float32(size(logits, 2))
end

# ── Train Step (Reactant-compatible) ──────────────────────────────────────────

function train_step(model, ps, st, x, y, lr)
    # Use Enzyme for gradients through Reactant tracing
    dps = Enzyme.make_zero(ps)
    _, loss = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        cross_entropy_loss,
        Active,
        Const(model),
        Duplicated(ps, dps),
        Const(st),
        Const(x),
        Const(y),
    )
    # SGD update: ps = ps - lr * dps
    ps_new = Lux.recursive_map((p, g) -> p .- lr .* g, ps, dps)
    return ps_new, loss
end

# ── Benchmark ─────────────────────────────────────────────────────────────────

function benchmark_cnn(;
    batch_size::Int=128,
    image_size::Int=64,
    num_warmup::Int=30,
    num_runs::Int=20,
    lr::Float32=0.01f0,
)
    println("CNN Performance Benchmark (Julia/Lux/Reactant)")
    println("Config: batch_size=$batch_size, image_size=$(image_size)x$(image_size)x3")
    println("Task: Forward + Backward pass with SGD\n")

    # Load plugin
    load_mlx_plugin!()

    # Build model
    model = SimpleCNN()
    rng = Random.Xoshiro(0)
    ps, st = Lux.setup(rng, model)
    ps = ps |> Lux.f32

    # Create random data (Lux uses WHCN format: Width x Height x Channels x Batch)
    x = randn(Float32, image_size, image_size, 3, batch_size)
    y = rand(1:10, batch_size)

    # Convert to Reactant arrays
    x_r = Reactant.to_rarray(x)
    y_r = Reactant.to_rarray(y)
    ps_r = Reactant.to_rarray(ps)
    st_r = Reactant.to_rarray(st)

    # Compile the train step
    println("Compiling train step...")
    t_compile_start = time()
    compiled_step = @compile train_step(model, ps_r, st_r, x_r, y_r, lr)
    t_compile = time() - t_compile_start
    println("Compilation done in $(round(t_compile; digits=1))s ✓\n")

    # Warmup
    println("Warming up ($num_warmup steps)...")
    for i in 1:num_warmup
        ps_r, loss = compiled_step(model, ps_r, st_r, x_r, y_r, lr)
    end
    println("Warmup done ✓\n")

    # Timed runs
    println("="^50)
    println("Timed runs")
    println("="^50)
    times = Float64[]
    for i in 1:num_runs
        t0 = time_ns()
        ps_r, loss = compiled_step(model, ps_r, st_r, x_r, y_r, lr)
        loss_val = Float64(loss)  # forces materialization (block until ready)
        t1 = time_ns()
        dt_ms = (t1 - t0) / 1e6
        push!(times, dt_ms)
        println("  Run $i: $(round(dt_ms; digits=2))ms (loss: $(round(loss_val; digits=4)))")
    end

    # Exclude first run (may include deferred compilation)
    steady = times[2:end]
    mean_t = mean(steady)
    std_t = std(steady)
    min_t = minimum(steady)
    println("\n  Mean: $(round(mean_t; digits=2))ms ± $(round(std_t; digits=2))ms  Min: $(round(min_t; digits=2))ms  (excl. first run)")

    println("\n" * "="^50)
    println("SUMMARY")
    println("="^50)
    println("Julia/Lux/Reactant/MLX: $(round(mean_t; digits=2))ms ± $(round(std_t; digits=2))ms")
end

# ── Main ──────────────────────────────────────────────────────────────────────

benchmark_cnn()

"""
MLP Training Benchmark (Julia)
Equivalent to benchmark_mlp.py using Lux, Enzyme, and Reactant on the MLX backend.

Usage:
  julia +1.11 benchmarks/benchmark_mlp.jl
"""

using Libdl, Reactant, Lux, Enzyme, Random, Statistics

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

    plugin_path = joinpath(@__DIR__, "..", "src", "jax_mlx", "libmlx_pjrt_plugin.dylib")
    if !isfile(plugin_path)
        plugin_path = joinpath(@__DIR__, "..", "build", "libmlx_pjrt_plugin.dylib")
    end
    @assert isfile(plugin_path) "Cannot find libmlx_pjrt_plugin.dylib at $plugin_path"

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

# ── MLP Model ─────────────────────────────────────────────────────────────────

function SimpleMLP()
    return Chain(
        Dense(2048, 8192, relu),
        Dense(8192, 4096, relu),
        Dense(4096, 4096, relu),
        Dense(4096, 2048, relu),
        Dense(2048, 1024, relu),
        Dense(1024, 10),
    )
end

# ── Loss Function ─────────────────────────────────────────────────────────────

function cross_entropy_loss(model, ps, st, x, y)
    logits, st_ = model(x, ps, st)
    log_probs = logits .- Lux.logsumexp(logits; dims=1)
    classes = reshape(1:10, 10, 1)
    labels = reshape(y, 1, :)
    one_hot = Float32.(classes .== labels)
    return -sum(one_hot .* log_probs) / Float32(size(logits, 2))
end

# ── Train Step ────────────────────────────────────────────────────────────────

function train_step(model, ps, st, x, y, lr)
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
    ps_new = Lux.recursive_map((p, g) -> p .- lr .* g, ps, dps)
    return ps_new, loss
end

# ── Benchmark ─────────────────────────────────────────────────────────────────

function benchmark_mlp(;
    batch_size::Int=2048,
    input_dim::Int=2048,
    num_warmup::Int=30,
    num_runs::Int=20,
    lr::Float32=0.01f0,
)
    println("MLP Training Benchmark (Julia/Lux/Reactant)")
    println("Config: batch_size=$batch_size, input_dim=$input_dim")
    println("Model: Dense(1024→8192→4096→4096→2048→1024→10) + ReLU")
    println("Task: Forward + Backward pass with SGD\n")

    load_mlx_plugin!()

    model = SimpleMLP()
    rng = Random.Xoshiro(0)
    ps, st = Lux.setup(rng, model)
    ps = ps |> Lux.f32

    x = randn(Float32, input_dim, batch_size)
    y = rand(1:10, batch_size)

    x_r = Reactant.to_rarray(x)
    y_r = Reactant.to_rarray(y)
    ps_r = Reactant.to_rarray(ps)
    st_r = Reactant.to_rarray(st)

    println("Compiling train step...")
    t_compile_start = time()
    compiled_step = @compile train_step(model, ps_r, st_r, x_r, y_r, lr)
    t_compile = time() - t_compile_start
    println("Compilation done in $(round(t_compile; digits=1))s ✓\n")

    println("Warming up ($num_warmup steps)...")
    for i in 1:num_warmup
        ps_r, loss = compiled_step(model, ps_r, st_r, x_r, y_r, lr)
    end
    println("Warmup done ✓\n")

    println("="^50)
    println("Timed runs")
    println("="^50)
    times = Float64[]
    for i in 1:num_runs
        t0 = time_ns()
        ps_r, loss = compiled_step(model, ps_r, st_r, x_r, y_r, lr)
        loss_val = Float64(loss)
        t1 = time_ns()
        dt_ms = (t1 - t0) / 1e6
        push!(times, dt_ms)
        println("  Run $i: $(round(dt_ms; digits=2))ms (loss: $(round(loss_val; digits=4)))")
    end

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

benchmark_mlp()

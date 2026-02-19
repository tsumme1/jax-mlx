"""
Transformer Encoder Training Benchmark (Julia)
Equivalent to benchmark_transformer.py using Lux, Enzyme, and Reactant on the MLX backend.
Uses real MultiHeadAttention for proper benchmark comparison.

Usage:
  julia +1.11 benchmarks/benchmark_transformer.jl
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

# ── Transformer Model using real MultiHeadAttention ──────────────────────────

# Manual LayerNorm that compiles under Reactant
# Normalizes over dim 1 (features in Julia column-major: d_model × seq_len × batch)
struct ManualLayerNorm{D} <: Lux.AbstractLuxLayer
    d_model::D
end

function Lux.initialparameters(rng::AbstractRNG, l::ManualLayerNorm)
    return (scale=ones(Float32, l.d_model, 1), bias=zeros(Float32, l.d_model, 1))
end

function (l::ManualLayerNorm)(x, ps, st)
    μ = mean(x; dims=1)
    σ2 = mean((x .- μ) .^ 2; dims=1)
    x_norm = (x .- μ) ./ sqrt.(σ2 .+ Float32(1e-5))
    return ps.scale .* x_norm .+ ps.bias, st
end

function TransformerBlock(d_model::Int, d_ff::Int, nheads::Int)
    return Chain(
        # Pre-norm self-attention with residual
        SkipConnection(
            Chain(
                ManualLayerNorm(d_model),
                MultiHeadAttention(d_model; nheads=nheads),
                WrappedFunction(x -> x[1]),
            ),
            +,
        ),
        # Pre-norm FFN with residual
        SkipConnection(
            Chain(
                ManualLayerNorm(d_model),
                Dense(d_model, d_ff, gelu),
                Dense(d_ff, d_model),
            ),
            +,
        ),
    )
end

function SimpleTransformer(; d_model=256, d_ff=512, num_layers=6, nheads=8, num_classes=10)
    layers = []
    # Input projection
    push!(layers, Dense(d_model, d_model))
    # Transformer blocks with real multi-head attention
    for _ in 1:num_layers
        push!(layers, TransformerBlock(d_model, d_ff, nheads))
    end
    # Mean pool over seq_len (dim 2), then classify
    # Input: (d_model, seq_len, batch) → mean → (d_model, 1, batch) → dropdims → (d_model, batch)
    push!(layers, WrappedFunction(x -> dropdims(mean(x; dims=2); dims=2)))
    push!(layers, Dense(d_model, num_classes))
    return Chain(layers...)
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

function benchmark_transformer(;
    batch_size::Int=128,
    seq_len::Int=64,
    d_model::Int=256,
    nheads::Int=8,
    num_warmup::Int=30,
    num_runs::Int=20,
    lr::Float32=0.01f0,
)
    println("Transformer Training Benchmark (Julia/Lux/Reactant)")
    println("Config: batch_size=$batch_size, seq_len=$seq_len, d_model=$d_model, nheads=$nheads")
    println("Model: 6× TransformerBlock(d=$d_model, $nheads heads, ff=512, MHA) → MeanPool → Dense(10)")
    println("Task: Forward + Backward pass with SGD\n")

    load_mlx_plugin!()

    model = SimpleTransformer(d_model=d_model, nheads=nheads)
    rng = Random.Xoshiro(0)
    ps, st = Lux.setup(rng, model)
    ps = ps |> Lux.f32

    # Lux format: (d_model, seq_len, batch)
    x = randn(Float32, d_model, seq_len, batch_size)
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

benchmark_transformer()

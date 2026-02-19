"""
RNN (GRU) Training Benchmark (Julia)
2-layer GRU → Dense(relu) → Dense(10) with SGD.
Uses GRUCell with manual unrolling. Matches JAX-MLX and native MLX architectures.

Usage:
  julia +1.11 benchmarks/benchmark_rnn.jl
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

# ── 2-Layer Manual GRU ────────────────────────────────────────────────────────
# Manually unrolls two GRUCells over timesteps.
# Reactant traces through the for loops → straight-line compute (no while_loop).

struct TwoLayerGRU{S,C1,C2,D1,D2} <: Lux.AbstractLuxContainerLayer{(:gru_cell1, :gru_cell2, :dense1, :dense2)}
    seq_len::S
    gru_cell1::C1
    gru_cell2::C2
    dense1::D1
    dense2::D2
end

function TwoLayerGRU(; input_dim=64, hidden_dim=512, seq_len=64, num_classes=10)
    return TwoLayerGRU(
        seq_len,
        GRUCell(input_dim => hidden_dim),
        GRUCell(hidden_dim => hidden_dim),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, num_classes),
    )
end

function (m::TwoLayerGRU)(x, ps, st)
    # x: (input_dim, seq_len, batch)
    batch_size = size(x, 3)

    # --- Layer 1: GRU over input ---
    x_t = x[:, 1, :]  # (input_dim, batch)
    (h1, (h1_state,)), st_gru1 = m.gru_cell1(x_t, ps.gru_cell1, st.gru_cell1)

    # Collect all hidden states for layer 2
    h1_all = similar(h1, size(h1, 1), m.seq_len, batch_size)
    h1_all[:, 1, :] = h1

    for t in 2:m.seq_len
        x_t = x[:, t, :]
        (h1, (h1_state,)), st_gru1 = m.gru_cell1((x_t, (h1_state,)), ps.gru_cell1, st_gru1)
        h1_all[:, t, :] = h1
    end

    # --- Layer 2: GRU over layer 1's hidden states ---
    x2_t = h1_all[:, 1, :]
    (h2, (h2_state,)), st_gru2 = m.gru_cell2(x2_t, ps.gru_cell2, st.gru_cell2)

    for t in 2:m.seq_len
        x2_t = h1_all[:, t, :]
        (h2, (h2_state,)), st_gru2 = m.gru_cell2((x2_t, (h2_state,)), ps.gru_cell2, st_gru2)
    end

    # Classification head on final hidden state
    y, st_d1 = m.dense1(h2, ps.dense1, st.dense1)
    y, st_d2 = m.dense2(y, ps.dense2, st.dense2)
    return y, (gru_cell1=st_gru1, gru_cell2=st_gru2, dense1=st_d1, dense2=st_d2)
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

function benchmark_rnn(;
    batch_size::Int=128,
    seq_len::Int=64,
    input_dim::Int=64,
    hidden_dim::Int=512,
    num_warmup::Int=30,
    num_runs::Int=20,
    lr::Float32=0.001f0,
)
    println("RNN (GRU) Training Benchmark (Julia/Lux/Reactant)")
    println("Config: batch_size=$batch_size, seq_len=$seq_len, input_dim=$input_dim, hidden=$hidden_dim")
    println("Model: 2× GRU($input_dim→$hidden_dim, $seq_len steps, unrolled) → Dense($hidden_dim, relu) → Dense(10)")
    println("Task: Forward + Backward pass with SGD\n")

    load_mlx_plugin!()

    model = TwoLayerGRU(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len)
    rng = Random.Xoshiro(0)
    ps, st = Lux.setup(rng, model)
    ps = ps |> Lux.f32

    # Lux format: (input_dim, seq_len, batch)
    x = randn(Float32, input_dim, seq_len, batch_size)
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

benchmark_rnn()

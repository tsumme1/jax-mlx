# JAX MLX

A PJRT plugin enabling JAX to use Apple's MLX framework as a backend on Apple Silicon Macs.

## Status

✅ **362 ops tested and passing**


## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4)
- **Python:** 3.11+
- **Dependencies:** jax, jaxlib, mlx

## Installation

```bash
# Install build dependencies
pip install mlx jaxlib jax

# Install the plugin
pip install .
```

## Usage

```python
import jax
import jax.numpy as jnp

# List available devices
print(jax.devices())  # [mlx:0]

# Use MLX as default device
mlx = jax.devices('mlx')[0]
with jax.default_device(mlx):
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sin(x) + jnp.cos(x)
    print(y)
```

## Features

- ✅ All core JAX operations (362 tested)
- ✅ Full autodiff support (`jax.grad`, `jax.value_and_grad`)
- ✅ JIT compilation with `mx.compile()` kernel fusion
- ✅ Vectorization (`jax.vmap`)
- ✅ Control flow (`lax.cond`, `lax.while_loop`, `lax.scan`)
- ✅ Linear algebra, FFT, convolutions
- ✅ Neural network training (Flax, Optax)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MLX_PJRT_DEBUG=1` | Enable verbose debug logging |
| `MLX_NO_COMPILE=1` | Disable mx.compile() kernel fusion |
| `MLX_TIMING=1` | Enable timing output |

## Development

```bash
# Run exhaustive tests (362 ops)
python tests/test_exhaustive.py

# Run CNN benchmarks
python benchmarks/benchmark_cnn.py           # JAX/Flax
python benchmarks/benchmark_mlx_native.py    # Native MLX Python

# Build and run C++ benchmark
cd benchmarks && cmake -B build && cmake --build build
./build/mlx_cpp_benchmark
```

## Architecture

The plugin implements the PJRT (Portable JAX Runtime) C API. StableHLO operations from JAX are parsed and converted to MLX operations at runtime using a lightweight MLIR parser. The plugin uses `mx.compile()` for GPU kernel fusion.

## Known Limitations

- **Float64:** Not supported on Metal GPU (use Float32)
- **While loops:** Block kernel fusion (require runtime eval)

## License

MIT License - see [LICENSE](LICENSE) for details.

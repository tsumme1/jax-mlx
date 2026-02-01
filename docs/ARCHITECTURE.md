# Architecture

## Overview

The JAX MLX plugin implements the PJRT (Portable JAX Runtime) C API, allowing JAX to dispatch computations to Apple's MLX framework.

## Components

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   JAX / Flax    │────▶│  PJRT Plugin     │────▶│   MLX Metal     │
│                 │     │  (C++ / MLIR)    │     │   GPU Kernels   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `src/jax_mlx_pjrt.cpp` | Main PJRT implementation (~300KB) |
| `src/mlx_mlir_parser.h` | Lightweight StableHLO parser |
| `src/mlx_pjrt_types.h` | Buffer and device types |
| `src/jax_mlx/` | Python package |

## Execution Flow

1. **JAX compiles** to StableHLO bytecode
2. **Plugin parses** bytecode using lightweight MLIR parser
3. **Operations mapped** to MLX C++ primitives
4. **`mx.compile()`** fuses operations into Metal kernels
5. **Results returned** via unified memory (zero-copy)

## Memory Model

MLX uses unified memory on Apple Silicon - no explicit host/device transfers needed. Buffers are shared between CPU and GPU.

## Supported Operations

All major StableHLO ops including:
- Element-wise: add, multiply, exp, sin, cos, etc.
- Reductions: sum, max, mean, etc.
- Linear algebra: matmul, dot, transpose, etc.
- Convolutions: conv_general_dilated
- Control flow: cond, while_loop, scan
- RNG: uniform, normal distributions

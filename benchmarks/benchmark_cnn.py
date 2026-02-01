"""
CNN Performance Comparison: CPU vs MLX
Simple CNN forward/backward pass benchmark using vanilla SGD.
"""

import os
# Disable PJRT debug logging for performance benchmark
if "MLX_PJRT_DEBUG" in os.environ:
    del os.environ["MLX_PJRT_DEBUG"]

import jax
import jax.numpy as jnp
from flax import linen as nn
import time
import numpy as np


class SimpleCNN(nn.Module):
    """Deep CNN for benchmarking."""
    
    @nn.compact
    def __call__(self, x):
        # Block 1
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Block 2
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Block 3
        x = nn.Conv(features=256, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


def loss_fn(params, model, x, y):
    """Cross-entropy loss."""
    logits = model.apply(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))


def create_train_step(model, lr=0.01):
    """Create a JIT-compiled train step."""
    @jax.jit
    def train_step(params, x, y):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, model, x, y))(params)
        # Simple SGD update
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        return params, loss
    return train_step


def benchmark_device(device, batch_size=256, image_size=64, num_warmup=15, num_runs=10):
    """Benchmark forward/backward pass on device."""
    print(f"\n{'='*50}")
    print(f"Benchmarking: {device.platform.upper()}")
    print(f"{'='*50}")
    
    model = SimpleCNN()
    input_shape = (batch_size, image_size, image_size, 3)
    
    with jax.default_device(device):
        # Initialize on this device
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.ones(input_shape))
        train_step = create_train_step(model)
        
        # Create dummy data
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, input_shape)
        y = jax.random.randint(key, (batch_size,), 0, 10)
        
        # Warmup
        print(f"Warmup ({num_warmup} runs)...")
        for _ in range(num_warmup):
            params, loss = train_step(params, x, y)
            loss.block_until_ready()
        
        # Timed runs
        print(f"Benchmarking ({num_runs} runs)...")
        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            params, loss = train_step(params, x, y)
            loss.block_until_ready()
            end = time.perf_counter()
            times.append(end - start)
            print(f"  Run {i+1}: {times[-1]*1000:.2f}ms (loss: {float(loss):.4f})")
    
    mean_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    
    print(f"\nResults for {device.platform.upper()}:")
    print(f"  Mean: {mean_time:.2f}ms ± {std_time:.2f}ms")
    print(f"  Min:  {min(times)*1000:.2f}ms")
    print(f"  Max:  {max(times)*1000:.2f}ms")
    
    return mean_time, std_time


def main():
    print("CNN Performance Benchmark: CPU vs MLX")
    print("Task: Forward + Backward pass with SGD")
    
    cpu = jax.devices("cpu")[0]
    try:
        mlx = jax.devices("mlx")[0]
    except Exception as e:
        print(f"MLX device not available: {e}")
        return
    
    
    batch_size = 256
    image_size = 64
    
    print(f"\nConfig: batch_size={batch_size}, image_size={image_size}x{image_size}x3")
    
    cpu_mean, cpu_std = benchmark_device(cpu, batch_size, image_size)
    mlx_mean, mlx_std = benchmark_device(mlx, batch_size, image_size)
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"CPU: {cpu_mean:.2f}ms ± {cpu_std:.2f}ms")
    print(f"MLX: {mlx_mean:.2f}ms ± {mlx_std:.2f}ms")
    
    if cpu_mean > 0:
        if mlx_mean < cpu_mean:
            print(f"\nMLX is {cpu_mean/mlx_mean:.2f}x FASTER than CPU")
        else:
            print(f"\nMLX is {mlx_mean/cpu_mean:.2f}x SLOWER than CPU")
    else:
        print("\nSkipped CPU benchmark comparison.")


if __name__ == "__main__":
    main()

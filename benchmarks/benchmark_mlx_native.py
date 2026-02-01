import mlx.core as mx
import mlx.nn as nn
import time
import numpy as np

# ============================================================
# Functional approach - params passed explicitly for proper compile
# ============================================================

def conv2d(x, w, b, stride=1, padding=1):
    """Conv2d with bias"""
    out = mx.conv2d(x, w, stride=stride, padding=padding)
    return out + b

def relu(x):
    return mx.maximum(x, 0)

def maxpool2d(x):
    """2x2 max pooling"""
    N, H, W, C = x.shape
    x = mx.reshape(x, (N, H//2, 2, W//2, 2, C))
    return mx.max(x, axis=(2, 4))

def cross_entropy(logits, labels):
    """Numerically stable cross-entropy"""
    max_logits = mx.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logits
    log_softmax = shifted - mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    indices = mx.expand_dims(labels, -1)
    log_probs = mx.take_along_axis(log_softmax, indices.astype(mx.int32), axis=1)
    return -mx.mean(log_probs)

def forward(x, params):
    """Forward pass through CNN"""
    c1w, c1b, c2w, c2b, c3w, c3b, c4w, c4b, c5w, c5b, d1w, d1b, d2w, d2b = params
    
    # Block 1
    h = relu(conv2d(x, c1w, c1b))
    h = relu(conv2d(h, c2w, c2b))
    h = maxpool2d(h)
    
    # Block 2
    h = relu(conv2d(h, c3w, c3b))
    h = relu(conv2d(h, c4w, c4b))
    h = maxpool2d(h)
    
    # Block 3
    h = relu(conv2d(h, c5w, c5b))
    
    # Global average pool + Dense
    h = mx.mean(h, axis=(1, 2))
    h = relu(mx.matmul(h, d1w) + d1b)
    h = mx.matmul(h, d2w) + d2b
    return h

def loss_fn(params, x, y):
    """Compute cross-entropy loss"""
    logits = forward(x, params)
    return cross_entropy(logits, y)

def benchmark_native(batch_size=256, image_size=64, num_warmup=5, num_runs=20):
    print(f"\n{'='*60}")
    print(f"Benchmarking: MLX Native (functional compiled)")
    print(f"{'='*60}")
    print(f"Config: batch_size={batch_size}, image_size={image_size}x{image_size}x3")

    lr = 0.01
    
    # Initialize weights (MLX conv2d format: [out_channels, kH, kW, in_channels])
    scale = (2.0 / (3*3*3)) ** 0.5
    c1w = mx.random.normal((64, 3, 3, 3)) * scale
    c1b = mx.zeros((64,))
    
    scale = (2.0 / (3*3*64)) ** 0.5
    c2w = mx.random.normal((64, 3, 3, 64)) * scale
    c2b = mx.zeros((64,))
    
    c3w = mx.random.normal((128, 3, 3, 64)) * scale
    c3b = mx.zeros((128,))
    
    scale = (2.0 / (3*3*128)) ** 0.5
    c4w = mx.random.normal((128, 3, 3, 128)) * scale
    c4b = mx.zeros((128,))
    
    c5w = mx.random.normal((256, 3, 3, 128)) * scale
    c5b = mx.zeros((256,))
    
    scale = (2.0 / 256) ** 0.5
    d1w = mx.random.normal((256, 128)) * scale
    d1b = mx.zeros((128,))
    
    scale = (2.0 / 128) ** 0.5
    d2w = mx.random.normal((128, 10)) * scale
    d2b = mx.zeros((10,))
    
    params = [c1w, c1b, c2w, c2b, c3w, c3b, c4w, c4b, c5w, c5b, d1w, d1b, d2w, d2b]
    mx.eval(params)
    
    # Dummy data
    x = mx.random.normal((batch_size, image_size, image_size, 3))
    y = mx.random.randint(0, 10, (batch_size,))
    mx.eval(x, y)
    
    # Create grad function
    grad_fn = mx.value_and_grad(loss_fn)
    
    # Full train step returns [loss, updated_params...]
    def train_step(params, x, y):
        loss, grads = grad_fn(params, x, y)
        new_params = [p - lr * g for p, g in zip(params, grads)]
        return [loss] + new_params
    
    # COMPILE the train step
    compiled_train_step = mx.compile(train_step)
    
    print(f"\nWarmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        result = compiled_train_step(params, x, y)
        mx.eval(result)
        params = result[1:]
    
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    losses = []
    
    for i in range(num_runs):
        start = time.perf_counter()
        result = compiled_train_step(params, x, y)
        mx.eval(result)
        params = result[1:]
        end = time.perf_counter()
        
        times.append((end - start) * 1000)
        losses.append(float(result[0]))
        
        if i < 5 or i >= num_runs - 5:
            print(f"  Run {i+1}: {times[-1]:.2f}ms, loss={losses[-1]:.4f}")
        elif i == 5:
            print("  ...")
    
    mean_time = np.mean(times[3:])
    std_time = np.std(times[3:])
    
    print(f"\n{'='*60}")
    print(f"Results for MLX Native (functional compiled):")
    print(f"{'='*60}")
    print(f"  Mean: {mean_time:.2f}ms ± {std_time:.2f}ms")
    print(f"  Min:  {min(times[3:]):.2f}ms")
    print(f"  Max:  {max(times[3:]):.2f}ms")
    print(f"\n  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Loss decreased: {'YES ✓' if losses[-1] < losses[0] else 'NO ✗'}")
    
    return mean_time

if __name__ == "__main__":
    benchmark_native()

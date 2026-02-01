/**
 * MLX C++ API Benchmark
 * 
 * Tests compiled forward + backward + parameter update (full training step)
 * with loss verification to confirm correct implementation.
 */

#include <mlx/mlx.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <functional>

namespace mx = mlx::core;

// Helper for ReLU
mx::array relu(const mx::array& x) {
    return mx::maximum(x, mx::array(0.0f));
}

// 2D Convolution (NHWC format) with proper API
mx::array conv2d_op(const mx::array& x, const mx::array& w, const mx::array& b, 
                 std::pair<int,int> stride = {1,1}, std::pair<int,int> padding = {1,1}) {
    auto out = mx::conv2d(x, w, stride, padding);
    return out + b;
}

// Max pooling 2x2
mx::array maxpool2d(const mx::array& x) {
    int N = x.shape()[0];
    int H = x.shape()[1];
    int W = x.shape()[2];
    int C = x.shape()[3];
    
    auto reshaped = mx::reshape(x, {N, H/2, 2, W/2, 2, C});
    auto pooled = mx::max(reshaped, std::vector<int>{2, 4});
    return pooled;
}

// Cross-entropy loss
mx::array cross_entropy(const mx::array& logits, const mx::array& labels) {
    auto max_logits = mx::max(logits, -1, true);
    auto shifted = logits - max_logits;
    auto log_softmax = shifted - mx::log(mx::sum(mx::exp(shifted), -1, true));
    
    int batch_size = logits.shape()[0];
    auto indices = mx::reshape(labels, {batch_size, 1});
    auto log_probs = mx::take_along_axis(log_softmax, indices, 1);
    
    return mx::negative(mx::mean(log_probs));
}

// Forward function that takes all inputs including weights
std::vector<mx::array> forward_with_loss(const std::vector<mx::array>& inputs) {
    // inputs[0] = x, inputs[1] = y, inputs[2...] = params
    const auto& x = inputs[0];
    const auto& y = inputs[1];
    
    int idx = 2;
    auto c1w = inputs[idx++]; auto c1b = inputs[idx++];
    auto c2w = inputs[idx++]; auto c2b = inputs[idx++];
    auto c3w = inputs[idx++]; auto c3b = inputs[idx++];
    auto c4w = inputs[idx++]; auto c4b = inputs[idx++];
    auto c5w = inputs[idx++]; auto c5b = inputs[idx++];
    auto d1w = inputs[idx++]; auto d1b = inputs[idx++];
    auto d2w = inputs[idx++]; auto d2b = inputs[idx++];
    
    // Forward pass
    auto h = relu(conv2d_op(x, c1w, c1b));
    h = relu(conv2d_op(h, c2w, c2b));
    h = maxpool2d(h);
    h = relu(conv2d_op(h, c3w, c3b));
    h = relu(conv2d_op(h, c4w, c4b));
    h = maxpool2d(h);
    h = relu(conv2d_op(h, c5w, c5b));
    h = mx::mean(h, std::vector<int>{1, 2});
    h = relu(mx::matmul(h, d1w) + d1b);
    h = mx::matmul(h, d2w) + d2b;
    
    auto loss = cross_entropy(h, y);
    return {loss};
}

int main() {
    std::cout << "=== MLX C++ API Benchmark ===" << std::endl;
    std::cout << "Compiled forward + backward + parameter update" << std::endl;
    
    const int batch_size = 256;
    const int img_size = 64;
    const int warmup_iters = 3;
    const int bench_iters = 10;
    const float lr = 0.01f;
    
    // Initialize weights (MLX conv2d format: [out_channels, kH, kW, in_channels])
    float scale = std::sqrt(2.0f / (3.0f * 3.0f * 3.0f));
    auto conv1_w = mx::random::normal({64, 3, 3, 3}) * scale;
    auto conv1_b = mx::zeros({64});
    
    scale = std::sqrt(2.0f / (3.0f * 3.0f * 64.0f));
    auto conv2_w = mx::random::normal({64, 3, 3, 64}) * scale;
    auto conv2_b = mx::zeros({64});
    
    auto conv3_w = mx::random::normal({128, 3, 3, 64}) * scale;
    auto conv3_b = mx::zeros({128});
    
    scale = std::sqrt(2.0f / (3.0f * 3.0f * 128.0f));
    auto conv4_w = mx::random::normal({128, 3, 3, 128}) * scale;
    auto conv4_b = mx::zeros({128});
    
    auto conv5_w = mx::random::normal({256, 3, 3, 128}) * scale;
    auto conv5_b = mx::zeros({256});
    
    scale = std::sqrt(2.0f / 256.0f);
    auto dense1_w = mx::random::normal({256, 128}) * scale;
    auto dense1_b = mx::zeros({128});
    
    scale = std::sqrt(2.0f / 128.0f);
    auto dense2_w = mx::random::normal({128, 10}) * scale;
    auto dense2_b = mx::zeros({10});
    
    // Materialize weights
    mx::eval({conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b,
              conv4_w, conv4_b, conv5_w, conv5_b, dense1_w, dense1_b,
              dense2_w, dense2_b});
    
    // Create input data
    auto x = mx::random::normal({batch_size, img_size, img_size, 3});
    auto y = mx::random::randint(0, 10, {batch_size});
    mx::eval({x, y});
    
    std::cout << "\nModel initialized. Input shape: [" << batch_size << ", " 
              << img_size << ", " << img_size << ", 3]" << std::endl;
    
    // Create params vector
    std::vector<mx::array> params = {
        conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b,
        conv4_w, conv4_b, conv5_w, conv5_b, dense1_w, dense1_b,
        dense2_w, dense2_b
    };
    
    // === Create grad function ===
    // Differentiate w.r.t. the weight parameters (indices 2 onwards)
    std::vector<int> grad_indices;
    for (int i = 2; i < 2 + 14; i++) {
        grad_indices.push_back(i);
    }
    
    auto grad_fn = mx::value_and_grad(forward_with_loss, grad_indices);
    
    // Create a wrapper that returns a flat vector (loss + all grads)
    // so it can be compiled
    auto train_step_fn = [&grad_fn, lr](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        auto [losses, grads] = grad_fn(inputs);
        
        // Return: [loss, updated_params...]
        std::vector<mx::array> result;
        result.push_back(losses[0]);
        
        // Apply SGD update and return new params
        for (size_t i = 0; i < grads.size(); i++) {
            result.push_back(inputs[2 + i] - lr * grads[i]);
        }
        return result;
    };
    
    // Compile the full train step
    auto compiled_train_step = mx::compile(train_step_fn);
    
    // === Warmup ===
    std::cout << "\nWarming up..." << std::endl;
    std::vector<mx::array> full_inputs = {x, y};
    for (auto& p : params) full_inputs.push_back(p);
    
    for (int i = 0; i < warmup_iters; i++) {
        auto results = compiled_train_step(full_inputs);
        mx::eval(results);
        // Update params in full_inputs
        for (size_t j = 0; j < params.size(); j++) {
            full_inputs[2 + j] = results[1 + j];
        }
    }
    
    // === Benchmark compiled forward+backward ===
    std::cout << "\n--- Compiled Forward + Backward + Update ---" << std::endl;
    
    std::vector<double> times;
    for (int i = 0; i < bench_iters; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto results = compiled_train_step(full_inputs);
        mx::eval(results);
        // Update params
        for (size_t j = 0; j < params.size(); j++) {
            full_inputs[2 + j] = results[1 + j];
        }
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double sq_sum = 0;
    for (auto t : times) sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / times.size());
    std::cout << "Compiled train step: " << mean << "ms +/- " << stddev << "ms" << std::endl;
    
    // === Full Training Loop with Loss Verification ===
    std::cout << "\n--- Training Loop (verifying loss decreases) ---" << std::endl;
    std::cout << "Training for 20 steps...\n" << std::endl;
    
    // Reset params
    full_inputs = {x, y};
    for (auto& p : params) full_inputs.push_back(p);
    
    std::vector<float> loss_history;
    times.clear();
    
    for (int step = 0; step < 20; step++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto results = compiled_train_step(full_inputs);
        mx::eval(results);
        
        // Update params
        for (size_t j = 0; j < params.size(); j++) {
            full_inputs[2 + j] = results[1 + j];
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(time_ms);
        
        // Get loss value
        float loss_val = results[0].item<float>();
        loss_history.push_back(loss_val);
        
        if (step < 5 || step >= 15) {
            std::cout << "Step " << step << ": loss = " << loss_val 
                      << ", time = " << time_ms << "ms" << std::endl;
        } else if (step == 5) {
            std::cout << "..." << std::endl;
        }
    }
    
    // Calculate average time (excluding first few warmup steps)
    double avg_time = 0;
    for (size_t i = 3; i < times.size(); i++) {
        avg_time += times[i];
    }
    avg_time /= (times.size() - 3);
    
    // Verify loss decreased
    float initial_loss = loss_history[0];
    float final_loss = loss_history.back();
    bool loss_decreased = final_loss < initial_loss;
    
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Initial loss: " << initial_loss << std::endl;
    std::cout << "Final loss:   " << final_loss << std::endl;
    std::cout << "Loss decreased: " << (loss_decreased ? "YES ✓" : "NO ✗") << std::endl;
    std::cout << "\nAverage step time: " << avg_time << "ms" << std::endl;
    
    return loss_decreased ? 0 : 1;
}

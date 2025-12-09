#include <iostream>
#include <vector>
#include <cuda_fp16.h>
#include "wrapper.cuh"

// Simple random generator
__half random_half() {
    float f = static_cast<float>(rand()) / RAND_MAX;
    return __float2half(f);
}

int main() {
    int B = 1;
    int H = 4;
    int L = 2048;
    int D = 128;

    int KQVO_block_y = 16;
    int warps = 8;

    size_t total = (size_t)B * H * L * D;
    std::vector<__half> hQ(total), hK(total), hV(total), hO(total);

    // fill input with random numbers
    for (size_t i = 0; i < total; i++) {
        hQ[i] = random_half();
        hK[i] = random_half();
        hV[i] = random_half();
    }

    float ms = 0.0f;

    run_cuda_cores_flash_attention_host_half(
        hQ.data(), hK.data(), hV.data(), hO.data(),
        B, H, L, D,
        KQVO_block_y, warps,
        0,     // cuda stream
        &ms    // record time
    );

    std::cout << "FlashAttention kernel time = " << ms << " ms\n";
    return 0;
}

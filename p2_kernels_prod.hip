// p2_kernels_prod.hip -- production M=1 W4A16 GEMV for gfx906, V3 structure.
// Scope (P1 report v3): whitelisted (K,N) = (5120,8192) & (5120,7168), M == 1.
// No torch headers: pure HIP + a C API driven by ctypes from the Python shim.
//   x    : fp16 [K]        (one decode row)
//   qw   : int32 [K/8, N]  (bridge layout, L0-pinned nibble order)
//   sc   : fp16 [K/128, N]
//   zr   : int32 [K/128, N/8]  (GPTQv2: stored nibble == true zp)
//   part : fp32 [splitk, N]   (caller-owned persistent scratch)
//   out  : fp16 [N]           (caller-owned capture-safe slab; OVERWRITE, no
//                              zero_ needed -- no atomicAdd anywhere)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>   // NOT <cuda_fp16.h>: that name does not exist on ROCm
                            // 7.2.4 (hipcc: file not found). P1 got away with it
                            // only because torch's load_inline adds shim -I paths;
                            // a standalone hipcc build has none. Container-verified
                            // 2026-09-09.

#define WAVE 64
#define GROUP 128
#define MAGIC 0x64006400u
#define UNROLL 8            // must divide GROUP/8 = 16; see the loop comment

__device__ __forceinline__ float dot2(float acc, unsigned a, unsigned b) {
  float r;
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(acc));
  return r;
}

// extern "C" stays on the HOST launcher only (ROCm HIP FAQ pattern). The
// __global__ kernels are launched through hipLaunchKernelGGL by function
// pointer inside this same .so -- they need no C linkage, and marking them
// extern "C" buys nothing (P2 review P1).
__global__ __launch_bounds__(WAVE) void p2_gemv_kernel(
    const __half* __restrict__ x, const int* __restrict__ qw,
    const __half* __restrict__ sc, const int* __restrict__ zr,
    float* __restrict__ part, int K, int N, int words_per_chunk) {
  __shared__ __half xs[GROUP];
  const int lane = threadIdx.x & (WAVE - 1);
  const int n = min(blockIdx.x * WAVE + lane, N - 1);   // clamp: never diverge-exit
  const int kend = min(blockIdx.y * words_per_chunk * 8 + words_per_chunk * 8, K);
  const int g0 = (blockIdx.y * words_per_chunk * 8) / GROUP;
  const int g1 = kend / GROUP;
  float acc = 0.f;

  for (int g = g0; g < g1; ++g) {
    for (int t = lane; t < GROUP; t += WAVE) xs[t] = x[g * GROUP + t];
    __syncthreads();
    const int zz = __ldg(&zr[g * (N / 8) + (n >> 3)]);
    const float zp = (float)((zz >> ((n & 7) * 4)) & 0xF);
    const __half2 z2b = __halves2half2(__float2half(1024.f + zp),
                                       __float2half(1024.f + zp));
    // SINGLE accumulator -- deliberately. Task 0 (2026-09-09) measured 4 chains
    // at the same grid points and multi-acc LOST 1.0-1.5 us everywhere (sk=8
    // 67.34->68.56, sk=10 60.51->61.52) while costing nothing in registers
    // (VGPR 21, occupancy 10 waves/SIMD = gfx906 max, 0 spills). The serial dot2
    // chain was already hidden by the other waves on the same SIMD.
    float raw = 0.f;
    // TWO-LEVEL loop: outer steps by UNROLL, inner has a COMPILE-TIME trip
    // count so `#pragma unroll` fully unrolls it. `#pragma unroll 8` on the
    // single 16-iteration loop instead cost 2.2-2.4 us (3.7-4.0%) on both
    // families, bit-identical output, 10/10 same-signed over 5 alternating
    // rounds per family with a within-arm spread <=0.06 us (p2_unroll_ab.py;
    // p2_body_ab.py is the control that swaps only the function body, which
    // rules out the build flags). Task 0's "UNROLL is lazy" finding was true
    // only of THIS spelling: there UNROLL picked the inner width, not
    // whether the loop got unrolled at all.
    for (int kb = 0; kb < GROUP / 8; kb += UNROLL) {
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const unsigned w = (unsigned)__ldg(
            &qw[((g * GROUP / 8) + kb + u) * N + n]);
        const __half* xp = &xs[(kb + u) * 8];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
          const unsigned xv = *reinterpret_cast<const unsigned*>(xp + 2 * p);
          // pair p = (i2p, i2p+1) via ((w >> 4p) & 0x000F000F) + MAGIC
          // (L0-pinned: code i lives at bits (i//2)*4 + (i%2)*16)
          unsigned codes = ((w >> (4 * p)) & 0x000F000Fu) + MAGIC;
          const __half2 c = *reinterpret_cast<const __half2*>(&codes);
          const __half2 d = __hsub2(c, z2b);      // (1024+c)-(1024+zp) == c-zp
          codes = *reinterpret_cast<const unsigned*>(&d);
          raw = dot2(raw, codes, xv);
        }
      }
    }
    acc += __half2float(__ldg(&sc[g * N + n])) * raw;   // V3: codes are exact c-zp
    __syncthreads();
  }
  part[(long)blockIdx.y * N + n] = acc;
}

__global__ void p2_reduce(const float* __restrict__ part,
                          __half* __restrict__ out, int N, int splits) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float a = 0.f;
  for (int s = 0; s < splits; ++s) a += part[(long)s * N + n];
  out[n] = __float2half(a);
}

// Returns 0 on success, -1 on a rejected shape or a failed launch. The shim
// MUST surface a nonzero return: a silently dropped launch is the hardest
// failure class to debug post-capture (P2 review P1).
extern "C" int p2_gemv(const void* x, const void* qw, const void* sc,
                       const void* zr, void* part, void* out,
                       int K, int N, int splitk, void* stream) {
  if (K % GROUP != 0 || N % WAVE != 0) return -1;   // whitelisted shapes satisfy both
  const int words = K / 8;
  if (words % (int)splitk != 0) return -1;          // partial tail must not vanish
  const int wpb = words / (int)splitk;
  if (wpb * 8 % GROUP != 0) return -1;              // chunk must end on a group boundary
  dim3 grid(N / WAVE, (unsigned)splitk);
  hipStream_t s = (hipStream_t)stream;
  hipLaunchKernelGGL(p2_gemv_kernel, grid, dim3(WAVE), 0, s,
                     (const __half*)x, (const int*)qw, (const __half*)sc,
                     (const int*)zr, (float*)part, K, N, wpb);
  int nth = 256, nb = (N + nth - 1) / nth;
  hipLaunchKernelGGL(p2_reduce, dim3(nb), dim3(nth), 0, s,
                     (const float*)part, (__half*)out, N, (int)splitk);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

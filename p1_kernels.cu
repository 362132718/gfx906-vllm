// p1_kernels.cu -- prototype M=1 wave64 W4A16 GEMVs for L1. Measurement only:
// nothing here is meant to enter the delivery image, and P2 re-derives its own
// structure from whichever variant wins inside a CUDA graph.
//
// Bit mapping is the L0-proven one (docs/superpowers/plans/2026-09-08-p1-nibble-map.md):
// code i of word (kw, n) sits at bits (i//2)*4 + (i%2)*16, so pair p holds
// (i2p, i2p+1) reachable as one dword: ((w >> 4p) & 0x000F000F) + 0x64006400.
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

#define WAVE 64            // gfx906 is wave64. Not warpSize (not constexpr on ROCm 7).
#define GROUP 128
#define MAGIC 0x64006400u  // two fp16 1024.0 halves

__device__ __forceinline__ float dot2(float acc, unsigned a, unsigned b) {
  float r;
  // SI VOP3P: dst = src2 + src0[0]*src1[0] + src0[1]*src1[1] (fp16 pairs, fp32 acc).
  // Operand order verified against the gfx906 ISA reference in plan review.
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %3"
               : "=v"(r) : "v"(a), "v"(b), "v"(acc));
  return r;
}

// One wavefront owns 64 consecutive outputs (lane l -> n0+l) and walks K, so its
// 64 int32 loads of qweight[kw, n0..n0+63] are 256 B fully coalesced.
template <int UNROLL, int VARIANT>
__global__ __launch_bounds__(WAVE) void gemv(
    const __half* __restrict__ x, const int* __restrict__ qw,
    const __half* __restrict__ sc, const int* __restrict__ zr,
    float* __restrict__ part, int K, int N, int kper) {
  __shared__ __half xs[GROUP];
  __shared__ float Sg;
  __shared__ unsigned short half_lut[16];
  const int lane = threadIdx.x & (WAVE - 1);
  // Clamp, never `return`: a divergent exit before the in-loop __syncthreads is
  // undefined. All five real N are multiples of 64, so this is free here, but P2
  // must not inherit a half-full-block hang.
  const int n = min(blockIdx.x * WAVE + lane, N - 1);
  if (VARIANT == 2 && lane < 16) half_lut[lane] = __half_as_ushort(__float2half((float)lane));
  __syncthreads();
  const int kend = min(blockIdx.y * kper + kper, K);
  float acc = 0.f;

  for (int g = (blockIdx.y * kper) / GROUP; g < kend / GROUP; ++g) {
    for (int t = lane; t < GROUP; t += WAVE) xs[t] = x[g * GROUP + t];
    __syncthreads();

    // S_g is lane-independent and only V1 needs it (its codes carry a +1024 bias
    // that folds into the per-group tail). Butterfly: 2 cvt/add + 6 shfl_xor
    // ~= 16 wave slots. The plan's first draft had lane 0 serially summing 128
    // elements (~384 slots, more than the whole main body) -- that was the
    // "false kill" defect the external review caught.
    float sg = 0.f;
    if (VARIANT == 1) {
      float s = __half2float(xs[lane]) + __half2float(xs[lane + WAVE]);
      for (int m = 1; m < WAVE; m <<= 1) s += __shfl_xor(s, m);
      if (lane == 0) Sg = s;
      __syncthreads();
      sg = Sg;
    }

    const int zz = __ldg(&zr[g * (N / 8) + (n >> 3)]);
    const float zp = (float)((zz >> ((n & 7) * 4)) & 0xF);
    // V3 subtracts a BIASED zero point, so (1024+c)-(1024+zp) == c-zp exactly and
    // no per-group correction survives. (fp16 small ints are NOT 0x0000+c, so a
    // "codes without magic" V3 is unimplementable -- see review P1-1.)
    const __half2 z2b = __halves2half2(__float2half(1024.f + zp), __float2half(1024.f + zp));
    // V2's LUT codes are UNBIASED fp16(c), so it needs the plain zero point. Its
    // first build forgot this entirely -- the one-hot effective-code table read
    // as code+zp, i.e. V2 measured "sum(c*x)" with no zero point at all.
    const __half2 z2 = __halves2half2(__float2half(zp), __float2half(zp));
    float raw = 0.f;

    for (int kb = 0; kb < GROUP / 8; kb += UNROLL) {
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const unsigned w = (unsigned)__ldg(&qw[((g * GROUP / 8) + kb + u) * N + n]);
        const __half* xp = &xs[(kb + u) * 8];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
          const unsigned xv = *reinterpret_cast<const unsigned*>(xp + 2 * p);
          unsigned codes;
          if (VARIANT == 2) {
            // half_lut holds the fp16 BIT PATTERN of c, so the pair is just the
            // two halves combined and NO magic may be added afterwards (integer
            // add on packed fp16 is meaningless). On this interleaved layout the
            // design's "1 op/nibble LUT" degenerates: a pair's two nibbles sit 16
            // bits apart, so a byte-indexed 256-entry dword LUT cannot work and
            // this costs TWO ds_reads per pair (24 ops/word vs V1's 12); with the
            // zero point it is 28. What decides it is whether 64 lanes hitting 16
            // addresses bank-conflict.
            const unsigned lo = half_lut[(w >> (4 * p)) & 0xF];
            const unsigned hi = half_lut[(w >> (16 + 4 * p)) & 0xF];
            const __half2 c = __halves2half2(__ushort_as_half((unsigned short)lo),
                                             __ushort_as_half((unsigned short)hi));
            const __half2 d = __hsub2(c, z2);           // v_pk_sub_f16, 1 per pair
            codes = *reinterpret_cast<const unsigned*>(&d);
          } else {
            codes = ((w >> (4 * p)) & 0x000F000Fu) + MAGIC;
          }
          if (VARIANT == 3) {
            const __half2 c = *reinterpret_cast<const __half2*>(&codes);
            const __half2 d = __hsub2(c, z2b);          // v_pk_sub_f16, 1 per pair
            codes = *reinterpret_cast<const unsigned*>(&d);
          }
          raw = dot2(raw, codes, xv);
          if (VARIANT == 1) {
            // Debias per word: keeping 1024*x in the fp32 accumulator until the
            // group tail rounds at ~4096 scale and costs ~3e-3 relative on
            // random x (measured). One-hot x hid this completely -- the L0
            // trick cannot see accumulation noise, only wiring.
            const __half2 xh = *reinterpret_cast<const __half2*>(&xv);
            raw -= 1024.f * (__half2float(__low2half(xh)) + __half2float(__high2half(xh)));
          }
        }
      }
    }
    const float scale = __half2float(__ldg(&sc[g * N + n]));
    // V1 debiased per word above, so its tail only removes the hoisted zp*sg.
    // V2 (fp16(c) LUT) and V3 (biased-zp subtract) were already exact in-loop.
    acc += (VARIANT == 1) ? scale * (raw - zp * sg) : scale * raw;
    __syncthreads();
  }
  part[(long)blockIdx.y * N + n] = acc;
}

// Load-only probe (review P0-4): the SAME access pattern as gemv -- one wavefront
// owns 64 consecutive outputs and walks all its K words, 256 B per wavefront per
// step -- but NO nibble extraction, NO dot2, NO zp. XOR-accumulate keeps the
// loads alive; the store only fires on an impossible value so the compiler
// cannot delete the loop. Timing this against the qweight-only floor separates
// "the floor is not reachable" from "our kernel is bad": if load-only also sits
// 2-5.9x above the floor, the floor itself is missing a shape-dependent term and
// option C (tuning the losing families) loses its premise.
__global__ __launch_bounds__(WAVE) void p1_loadonly(
    const int* __restrict__ qw, unsigned* __restrict__ sink,
    int N, int words_per_chunk) {
  const int lane = threadIdx.x & (WAVE - 1);
  const int n = min(blockIdx.x * WAVE + lane, N - 1);
  unsigned acc = 0u;
  const int base = blockIdx.y * words_per_chunk;
  for (int i = 0; i < words_per_chunk; ++i)
    acc ^= (unsigned)__ldg(&qw[(long)(base + i) * N + n]);
  if (acc == 0xdeadbeefu) sink[blockIdx.y] = acc;
}

int64_t p1_loadonly_launch(torch::Tensor qw, torch::Tensor sink, int64_t splitk) {
  const int K = (int)qw.size(0) * 8, N = (int)qw.size(1);
  const int words = K / 8;
  TORCH_CHECK(words % (int)splitk == 0, "splitk must divide K/8");
  const int wpb = words / (int)splitk;
  const dim3 grid((N + WAVE - 1) / WAVE, (unsigned)splitk);
  p1_loadonly<<<grid, WAVE, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const int*)qw.data_ptr(), (unsigned*)sink.data_ptr(), N, wpb);
  return 0;
}

// Review P0-1: the scalar probe loads 4 B/lane with no unroll and a single XOR
// accumulator -- exactly the anti-pattern ROCm's tuning guide calls out
// ("ensure global_load_dwordx4 ... especially when the load happens in the
// loop"). Two fixed variants, both timed so saturation is answered by
// measurement rather than argument:
//   mlp : scalar dword, but 4 INDEPENDENT XOR accumulators over an unrolled 4x
//         loop (breaks the chain; still 4 B/lane per issue);
//   x4  : one dwordx4 (16 B) per lane covering 4 consecutive OUTPUTS at the
//         same k-word -- contiguous in n, the natural vectorization of this
//         layout; a wavefront covers 256 outputs (1 KiB) per step.
__global__ __launch_bounds__(WAVE) void p1_loadonly_mlp(
    const int* __restrict__ qw, unsigned* __restrict__ sink,
    int N, int words_per_chunk) {
  const int lane = threadIdx.x & (WAVE - 1);
  const int n = min(blockIdx.x * WAVE + lane, N - 1);
  unsigned a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
  const int base = blockIdx.y * words_per_chunk;
  int i = 0;
  for (; i + 4 <= words_per_chunk; i += 4) {
    a0 ^= (unsigned)__ldg(&qw[(long)(base + i + 0) * N + n]);
    a1 ^= (unsigned)__ldg(&qw[(long)(base + i + 1) * N + n]);
    a2 ^= (unsigned)__ldg(&qw[(long)(base + i + 2) * N + n]);
    a3 ^= (unsigned)__ldg(&qw[(long)(base + i + 3) * N + n]);
  }
  for (; i < words_per_chunk; ++i)
    a0 ^= (unsigned)__ldg(&qw[(long)(base + i) * N + n]);
  if ((a0 ^ a1 ^ a2 ^ a3) == 0xdeadbeefu) sink[blockIdx.y] = a0;
}

__global__ __launch_bounds__(WAVE) void p1_loadonly_x4(
    const int4* __restrict__ qw4, unsigned* __restrict__ sink,
    int N4, int v4_per_chunk) {
  // N4 = N/4 int4-vectors per word-row; lane l covers outputs 4l..4l+3.
  const int lane = threadIdx.x & (WAVE - 1);
  const int n4 = min(blockIdx.x * WAVE + lane, N4 - 1);
  int4 acc = make_int4(0, 0, 0, 0);
  const int base = blockIdx.y * v4_per_chunk;
  for (int i = 0; i < v4_per_chunk; ++i) {
    const int4 v = __ldg(&qw4[(long)(base + i) * N4 + n4]);
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((unsigned)(acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xdeadbeefu)
    sink[blockIdx.y] = acc.x;
}

int64_t p1_loadonly_v2_launch(torch::Tensor qw, torch::Tensor sink,
                              int64_t splitk, int64_t kind) {
  const int K = (int)qw.size(0) * 8, N = (int)qw.size(1);
  const int words = K / 8;
  TORCH_CHECK(words % (int)splitk == 0, "splitk must divide K/8");
  const int wpb = words / (int)splitk;
  if (kind == 0) {
    const dim3 grid((N + WAVE - 1) / WAVE, (unsigned)splitk);
    p1_loadonly_mlp<<<grid, WAVE, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const int*)qw.data_ptr(), (unsigned*)sink.data_ptr(), N, wpb);
    return 0;
  }
  TORCH_CHECK(N % 4 == 0, "x4 probe needs N divisible by 4");
  const int N4 = N / 4;
  const dim3 grid((N4 + WAVE - 1) / WAVE, (unsigned)splitk);
  p1_loadonly_x4<<<grid, WAVE, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const int4*)qw.data_ptr(), (unsigned*)sink.data_ptr(), N4, wpb);
  return 0;
}

// Two-phase split-K reduction; also the float->half conversion for splitk==1.
__global__ void p1_reduce(const float* __restrict__ part, __half* __restrict__ out,
                          int N, int splits) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float a = 0.f;
  for (int s = 0; s < splits; ++s) a += part[(long)s * N + n];
  out[n] = __float2half(a);
}

torch::Tensor p1_gemv(torch::Tensor x, torch::Tensor qw, torch::Tensor sc,
                      torch::Tensor zr, torch::Tensor scratch, int64_t variant,
                      int64_t splitk, int64_t unroll, bool reduce) {
  const int K = (int)qw.size(0) * 8, N = (int)qw.size(1);
  const int kper = ((K / GROUP) / (int)splitk) * GROUP;
  TORCH_CHECK(kper >= GROUP, "splitk ", splitk, " leaves an empty K chunk for K=", K);
  TORCH_CHECK((K / GROUP) % (int)splitk == 0, "splitk ", splitk,
              " does not divide the ", K / GROUP, " groups of K=", K,
              " -- the tail would be silently dropped");
  auto out = torch::empty({N}, x.options().dtype(torch::kHalf));
  float* part = scratch.data_ptr<float>();
  const unsigned out_blocks = (N + WAVE - 1) / WAVE;
  const dim3 grid(out_blocks, (unsigned)splitk);   // x: outputs/64, y: split-K chunks.
  // Two launch bugs in a row here, both caught by the poison probe instead of by
  // reading: (1) <<<gx, WAVE>>> with a 1-D gx never ran chunks >= 1; (2) the first
  // "fix" passed gy.y, but dim3(splitk) stores the count in .x, so that was still
  // a y=1 launch. Verify what blocks ACTUALLY ran; do not trust the launch args.
  // Launch on TORCH'S CURRENT STREAM, not the legacy default: torch.cuda.graph
  // captures on a side stream, and default-stream launches are simply absent
  // from the captured graph -- replay then does nothing (measured 1.03 us for
  // every variant, far below the 10-57 us floors). This one bit the whole sweep.
  #define GO(V, U) gemv<U, V><<<grid, WAVE, 0, at::cuda::getCurrentCUDAStream()>>>( \
      (const __half*)x.data_ptr(), (const int*)qw.data_ptr(), \
      (const __half*)sc.data_ptr(), (const int*)zr.data_ptr(), part, K, N, kper)
  #define FOR(V) do { if (unroll == 4) GO(V, 4); else if (unroll == 8) GO(V, 8); else GO(V, 16); } while (0)
  if (variant == 1) FOR(1); else if (variant == 2) FOR(2); else FOR(3);
  #undef FOR
  #undef GO
  hipError_t lerr = hipGetLastError();
  TORCH_CHECK(lerr == hipSuccess, "gemv launch failed: ", hipGetErrorString(lerr));
  if (!reduce) {
    // Timing-only arm: returns the float partials, never pass it to check().
    // splitk=1 with vs without the reduce launch isolates the DEPENDENT reduce
    // cost -- two independent noops would let the graph overlap them.
    return scratch.view({(long)splitk, (long)N});
  }
  p1_reduce<<<(N + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      part, (__half*)out.data_ptr(), N, (int)splitk);
  // NOTE for P2: this helper allocates `out` per call; torch's graph pool absorbs
  // that during capture, but the production kernel must keep writing into the
  // caller-owned capture-safe slab. Do not copy this allocation pattern.
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("p1_gemv", &p1_gemv);
  m.def("p1_loadonly_launch", &p1_loadonly_launch);
  m.def("p1_loadonly_v2_launch", &p1_loadonly_v2_launch);
}

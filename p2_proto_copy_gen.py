# p2_proto_copy_gen.py -- generate p2_proto_copy.hip: the prototype's gemv kernel
# BODY (verbatim, extracted from p1_kernels.cu) inside the PRODUCTION launcher and
# the PRODUCTION build path (standalone hipcc).
#
# Why this exists: the production .so is bit-identical to the prototype yet 3.7-
# 4.0% slower, reproducibly. Two candidate causes remain and nothing in the source
# comments distinguishes them:
#   (a) source differences inside the kernel body (extra __shared__ decls, the
#       entry __syncthreads, the template's dead VARIANT branches, loop form),
#   (b) the build path (torch load_inline flags vs bare `hipcc -O3`).
# This file holds (b) fixed and swaps only the body, so one timing separates them:
#   ~60.4 us  -> (a): the body is what matters, bisect it next
#   ~62.8 us  -> (b): the body is innocent, go compare compiler flags
import sys

proto_cu, prod_hip, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

src = open(proto_cu, encoding="utf-8").read()
i = src.index("template <int UNROLL, int VARIANT>")
end = "part[(long)blockIdx.y * N + n] = acc;\n}\n"
j = src.index(end, i) + len(end)
body = src[i:j]
assert "__launch_bounds__(WAVE)" in body and "xs[t] = x[g * GROUP + t]" in body

prod = open(prod_hip, encoding="utf-8").read()
head = prod[:prod.index("__global__ __launch_bounds__(WAVE) void p2_gemv_kernel(")]
reduce_and_tail = prod[prod.index("__global__ void p2_reduce("):]
reduce_and_tail = reduce_and_tail[:reduce_and_tail.index("extern \"C\" int p2_gemv(")]

launcher = """
// Same grid, same chunk math, same caller-owned out, same reduce as production --
// ONLY the kernel above differs. kper is the prototype's per-chunk ELEMENT count.
extern "C" int p2_gemv_proto(const void* x, const void* qw, const void* sc,
                             const void* zr, void* part, void* out,
                             int K, int N, int splitk, void* stream) {
  const int words = K / 8;
  const int wpb = words / splitk;
  dim3 grid(N / WAVE, (unsigned)splitk);
  hipStream_t s = (hipStream_t)stream;
  gemv<8, 3><<<grid, WAVE, 0, s>>>((const __half*)x, (const int*)qw,
                                   (const __half*)sc, (const int*)zr,
                                   (float*)part, K, N, wpb * 8);
  int nth = 256, nb = (N + nth - 1) / nth;
  hipLaunchKernelGGL(p2_reduce, dim3(nb), dim3(nth), 0, s,
                     (const float*)part, (__half*)out, N, (int)splitk);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}
"""

open(out_path, "w", encoding="utf-8").write(
    head + body + "\n" + reduce_and_tail + launcher)
print("wrote %s: prototype body %d B + production head/tail" % (out_path, len(body)))
print("body first line: %s" % body.split("\n")[1][:70])

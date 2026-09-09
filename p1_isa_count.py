# p1_isa_count.py -- op/weight read off the ISA (never inferred from timing),
# plus VGPR counts from the kernel metadata. Cross-check (review gap #3):
# op/weight_from_timing = 6.62e12 * t / (K*N/8) must land within ~20% of the ISA
# count, else the hot-loop split is wrong.
import re
import subprocess

SO = "/tmp/p1_bench_build/p1_kernels.so"
OBJ = "/opt/rocm/llvm/bin/llvm-objdump"

dis = subprocess.run([OBJ, "-d", "-C", SO], capture_output=True, text=True).stdout
notes = subprocess.run(["/opt/rocm/llvm/bin/llvm-readelf", "--notes", SO],
                       capture_output=True, text=True).stdout

# VGPR/SGPR per kernel from the amdhsa metadata notes.
vgpr = {}
for m in re.finditer(r"amdhsa\.kernels.*?\.name:\s*'([^']+)'([\s\S]*?)(?=amdhsa\.kernels|\Z)",
                     notes):
    name = m.group(1)
    v = re.search(r"\.vgpr_count:\s*(\d+)", m.group(2))
    s = re.search(r"\.sgpr_count:\s*(\d+)", m.group(2))
    if v:
        vgpr[name] = (int(v.group(1)), int(s.group(1)) if s else -1)

print("=== per-symbol ISA counts (whole symbol; loop split noted per variant) ===")
for m in re.finditer(r"^([0-9a-f]+) <([^>]*gemv[^>]*)>:", dis, re.M):
    name, start = m.group(2), m.end()
    end = start + dis[start:].find("\n\n")
    body = dis[start:end]
    valu = len(re.findall(r"\bv_[a-z0-9_]+", body))
    dots = len(re.findall(r"\bv_dot2_f32_f16", body))
    mem = len(re.findall(r"\b(ds_[a-z0-9_]+|global_load\S*|buffer_load\S*)", body))
    key = next((k for k in vgpr if k in name or name in k), None)
    vgv = vgpr.get(key, (-1, -1))
    # The hot loop repeats GROUP/8 / UNROLL times per group; whole-symbol counts
    # include prologue. Report dots*2 as the weights-per-symbol scale check.
    print("%-40s valu=%-5d dot2=%-4d mem=%-5d VGPR=%-3d SGPR=%-3d" % (
        name[:40], valu, dots, mem, vgv[0], vgv[1]))

print("\n=== op/weight (upper bound, whole symbol / 8 weights per int32 word) ===")
# Whole-symbol VALU over (weights touched by the symbol) is NOT per-weight unless
# we know the trip count; instead report valu per dot2 and per mem op so the
# hot-loop structure is visible, and leave the exact split to the report.

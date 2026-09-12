#!/bin/bash
# p2_gate_teeth2.sh -- the mutation M1 could not test.
#
# M1 (swap w_s/w_zp) did NOT produce a silent numeric error: it produced
# "Memory access fault by GPU node-1 ... Reason: Page not present" and killed
# the process inside row A, because the kernel reads [K/128, N] fp16 of scales
# (655 KB at N=8192) out of what is really the qzeros buffer (163 KB). So M1
# disproved part of the gate header: that particular mistake is loud, not
# plausible-garbage.
#
# What row B's bit-equality really has to be able to catch is a mistake that is
# memory-safe AND silent -- you get a plausible tensor back, just the wrong
# number. M1' is exactly that: pass K/2 to the kernel. Every read stays inside
# its allocation (x, qw, sc, zr are all larger than what a K/2 walk touches),
# splitk still divides K/2/8, and the output is a partial reduction -- right
# shape, finite, plausible, wrong. If G-P2 stays green on this, row B is
# decorative and the gate has to be rebuilt.
set -u
D=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py
SRC=/work/p2/wna16_exllama.py
OUT=/work/p2/teeth2.log
: > $OUT
cp $SRC $D
rm -f $(dirname $D)/__pycache__/wna16_exllama*.pyc
sed -i 's/int(x\.shape\[1\]), int(out\.shape\[1\])/int(x.shape[1]) \/\/ 2, int(out.shape[1])/' $D
if grep -q 'int(x.shape\[1\]) // 2' $D; then MOD="M1p K passed to the kernel halved (silent, memory-safe)"; else MOD="M1p NOT APPLIED"; fi
echo "=========== $MOD ===========" >> $OUT
( cd /work/p2 && python3 -u p2_route_gate.py >> $OUT 2>&1 )
echo "SUMMARY :: $(grep -E '^G-P2|^ROUTE' $OUT | tr '\n' ' ') BAD_rows=$(grep -c 'BAD' $OUT)" >> $OUT
cp $SRC $D
rm -f $(dirname $D)/__pycache__/wna16_exllama*.pyc
md5sum $D $SRC >> $OUT
echo "TEETH2-DONE" >> $OUT

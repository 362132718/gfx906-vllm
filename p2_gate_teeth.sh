#!/bin/bash
# p2_gate_teeth.sh -- does p2_route_gate.py actually fail when the code under
# test is wrong? A gate that only ever prints ok proves nothing, so inject three
# specific defects, one per claimed weak point, and require each to turn a
# specific row red. After each run the pristine module is restored from
# /work/p2/wna16_exllama.py (the copy of container_src/wna16_exllama.py).
#
#   M1  swap w_s / w_zp in _p2_w4a16_gemm_impl. The plan's claim is that NO
#       numeric band catches this (both orders yield plausible garbage), so
#       row B's bit-equality-vs-ctypes must be the thing that catches it.
#   M2  drop the whitelist membership test from _p2_layer_ok. Rows A, D, F must
#       go red -- they are the predicate-precision assertions.
#   M3  remove "out" from mutates_args. The plan says functionalization then
#       drops the overwrite and the output stays zero. Whether that is really
#       how it fails is the question; either red row or a still-green gate is
#       the answer, and the answer goes in the record either way.
set -u
D=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py
SRC=/work/p2/wna16_exllama.py
OUT=/work/p2/teeth.log
: > $OUT
: > /work/p2/teeth_summary.log

for m in 1 2 3; do
  cp $SRC $D
  rm -f $(dirname $D)/__pycache__/wna16_exllama*.pyc
  case $m in
    1) sed -i 's/x\.data_ptr(), w_q\.data_ptr(), w_s\.data_ptr(), w_zp\.data_ptr(),/x.data_ptr(), w_q.data_ptr(), w_zp.data_ptr(), w_s.data_ptr(),/' $D
       grep -q 'w_zp.data_ptr(), w_s.data_ptr(),' $D && MOD="M1 w_s<->w_zp swapped in the impl" || MOD="M1 NOT APPLIED";;
    2) sed -i 's/and (int(K_dim), int(N_dim)) in _P2_SHAPES/and True/' $D
       grep -q 'and True' $D && MOD="M2 whitelist membership dropped" || MOD="M2 NOT APPLIED";;
    3) sed -i 's/mutates_args=\["part", "out"\]/mutates_args=["part"]/' $D
       grep -q 'mutates_args=\["part"\]' $D && MOD="M3 out removed from mutates_args" || MOD="M3 NOT APPLIED";;
  esac
  echo "=========== $MOD ===========" >> $OUT
  ( cd /work/p2 && python3 -u p2_route_gate.py > /work/p2/teeth_run.log 2>&1 )
  cat /work/p2/teeth_run.log >> $OUT
  echo "SUMMARY $MOD :: $(grep -E '^G-P2|^ROUTE' /work/p2/teeth_run.log | tr '\n' ' ') " \
       "BAD_rows=$(grep -c 'BAD' /work/p2/teeth_run.log) " \
       "traceback=$(grep -c Traceback /work/p2/teeth_run.log)" >> /work/p2/teeth_summary.log
  echo "---" >> $OUT
done
cp $SRC $D
rm -f $(dirname $D)/__pycache__/wna16_exllama*.pyc
md5sum $D $SRC >> $OUT
cat /work/p2/teeth_summary.log >> $OUT
echo "TEETH-DONE" >> $OUT

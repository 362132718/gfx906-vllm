#!/bin/bash
python3 - <<'PY'
import inspect
import vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 as m
print("FILE:", m.__file__)
src = inspect.getsource(m)
for i, line in enumerate(src.splitlines(), 1):
    if any(k in line for k in ("def ", "weight_packed", "weight_scale",
                               "weight_zero_point", "input_dim", "output_dim",
                               "packed_dim", "permute_param_layout",
                               "scales", "qweight", "qzeros")):
        print(i, line)
PY

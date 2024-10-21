import torch
import torch.nn as nn
import apex
from torch.profiler import profile, record_function, ProfilerActivity
from apex.normalization import FusedLayerNorm
import parse_trace
import sys
import ck_cmd

run_ck = False if len(sys.argv) <= 1 else bool(sys.argv[1])
class LayerNormModel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormModel, self).__init__()
        self.layer_norm1 = nn.LayerNorm(num_features)

    def forward(self, x):
        return self.layer_norm1(x)
    
class LayerNormModelApex(nn.Module):
    def __init__(self, num_features):
        super(LayerNormModelApex, self).__init__()
        self.layer_norm = apex.normalization.FusedLayerNorm(num_features)

    def forward(self, x):
        return self.layer_norm(x)

def run_fw(shape):
    v = ['' for _ in range(5)]
    input_shape, _, norm_shape,_,_ = shape
    v[0] = 'x'.join([str(v) for v in input_shape])
    v[1] = 'x'.join([str(v) for v in norm_shape])
    model = LayerNormModel(norm_shape).eval().cuda().to(torch.bfloat16)
    model2 = LayerNormModelApex(norm_shape).eval().cuda().to(torch.bfloat16)
    input_tensor = torch.randn(*input_shape, device="cuda").to(torch.bfloat16)
    input_tensor2 = torch.randn(*input_shape, device="cuda").to(torch.bfloat16)
    def run():
        cache_flush1 = torch.randn(10000, 10000, requires_grad=True, device="cuda", dtype=torch.float32).to(torch.int32)
        v1 = model(input_tensor)
        cache_flush2 = torch.randn(10000, 10000, requires_grad=True, device="cuda", dtype=torch.float32).to(torch.int32)
        v2 = model2(input_tensor2)
        return v1 + v2

    #warmup
    for _ in range(5):
        run()

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(100):
            run()
        torch.cuda.synchronize()
    prof.export_chrome_trace("trace_temp.json")
    torch_time = parse_trace.parse_trace_json(
         "trace_temp.json", 
         "at::native::(anonymous namespace)::vectorized_layer_norm_kernel")
    apex_time = parse_trace.parse_trace_json(
         "trace_temp.json", 
         "void cuApplyLayerNorm<")
    
    s_mul = 1
    norm_mul = 1
    for x in norm_shape:
        norm_mul *= x
    for x in input_shape:
        s_mul *= x
    if run_ck:
        ck_time = ck_cmd.run('layernorm_fwd', s_mul / norm_mul, norm_mul)
    else:
        ck_time = 0
    v[2] = ck_time * 1000
    v[3] = apex_time
    v[4] = torch_time
    print(','.join([str(i) for i in v]))


shapes = (
[[4096,200,64],[],[64],[64],[]],
[[4096,24,64],[],[64],[64],[]],
[[4096,512],[],[512],[512],[]],
[[4096,160],[],[160],[160],[]],
[[4096,2048],[],[2048],[2048],[]],
[[4096,4096],[],[4096],[4096],[]],
[[4096,64,160],[],[160],[160],[]],
)

print("shape, norm_shape, ck, apex, torch")
for s in shapes:
    run_fw(s)




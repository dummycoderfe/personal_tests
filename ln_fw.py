import torch
import torch.nn as nn
import apex
from torch.profiler import profile, record_function, ProfilerActivity
from apex.normalization import FusedLayerNorm
import parse_trace
import sys
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

def run_bw(shape):
    input_shape, _, norm_shape,_,_ = shape
    model = LayerNormModel(norm_shape).eval().cuda().to(torch.bfloat16)
    model2 = LayerNormModelApex(norm_shape).eval().cuda().to(torch.bfloat16)
    input_tensor = torch.randn(*input_shape, device="cuda").to(torch.bfloat16)
    def run():
        return model(input_tensor)

    def run_apex():
        return model2(input_tensor)

    #warmup
    for _ in range(5):
        run()
        run_apex()

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
                run()
                torch.cuda.synchronize()
        for _ in range(20):
                run_apex()
                torch.cuda.synchronize()
    prof.export_chrome_trace("trace_temp.json")
    print(input_shape)
    print("torch fowward,", parse_trace.parse_trace_json(
         "trace_temp.json", 
         "at::native::(anonymous namespace)::vectorized_layer_norm_kernel"))
    print("apex fowward,",parse_trace.parse_trace_json(
         "trace_temp.json", 
         "void cuApplyLayerNorm<"))
shapes = (
[[4096,200,64],[],[64],[64],[]],
[[4096,24,64],[],[64],[64],[]],
[[4096,512],[],[512],[512],[]],
[[4096,160],[],[160],[160],[]],
[[4096,2048],[],[2048],[2048],[]],
[[4096,4096],[],[4096],[4096],[]],
[[4096,64,160],[],[160],[160],[]],
)
for s in shapes:
     run_bw(s)




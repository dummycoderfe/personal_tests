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
    input_shape, norm_shape = shape
    model = LayerNormModel(norm_shape).cuda().to(torch.bfloat16)
    model2 = LayerNormModelApex(norm_shape).cuda().to(torch.bfloat16)
    input_tensor = torch.randn(*input_shape, requires_grad=True, device="cuda").to(torch.bfloat16)
    def run():
        output = model(input_tensor)
        loss = output.max()  
        loss.backward() 
        print(loss) 

    def run_apex():
        output2 = model2(input_tensor)
        loss2 = output2.max() 
        loss2.backward() 
        print(loss2) 

    #warmup
    for _ in range(5):
        run()
        run_apex()

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
                run()
                torch.cuda.synchronize()
        for _ in range(10):
                run_apex()
                torch.cuda.synchronize()
    prof.export_chrome_trace("trace_temp.json")

    print("torch time:")
    print("gammabeta_grad,", parse_trace.parse_trace_json(
         "trace_temp.json", 
         "void at::native::(anonymous namespace)::cuComputePartGradGammaBeta"))
    print("input_grad,", parse_trace.parse_trace_json(
         "trace_temp.json", 
         "void at::native::(anonymous namespace)::cuComputeGradInput",
         "void at::native::(anonymous namespace)::layer_norm_grad_input_kernel"))
    print("apex time:")
    print("gammabeta_grad,", parse_trace.parse_trace_json(
         "trace_temp.json", 
         "void cuComputePartGradGammaBeta<"))
    print("input_grad,", parse_trace.parse_trace_json(
         "trace_temp.json", 
         "void cuComputeGradInput<"))
    
shapes = (
[[2048, 130, 128],[], [128], [],[]], 
[[2048, 152, 256],[], [152, 256], [],[]],
[[2048,256],[],[256],[256],[]],
[[2048,1024],[],[1024],[1024],[]],
[[2048,2048],[],[2048],[2048],[]],
[[2048,6016],[],[6016],[6016],[]],
[[2048,4800],[],[4800],[4800],[]],
[[2048,2304],[],[2304],[2304],[]],
[[2048,257,192],[],[192],[192],[]],
[[2048,8064],[],[8064],[8064],[]],
[[2048,10112],[],[10112],[10112],[]],
[[2048,12160],[],[12160],[12160],[]],
[[2048,14208],[],[14208],[14208],[]],
[[2048,16256],[],[16256],[16256],[]],
[[2048,18304],[],[18304],[18304],[]],
[[2048,20352],[],[20352],[20352],[]],
[[2048,22400],[],[22400],[22400],[]],
[[2048,24448],[],[24448],[24448],[]],
[[2048,26496],[],[26496],[26496],[]],
[[2048,200,384],[],[384],[384],[]],
[[2048,200,256],[],[256],[256],[]],
[[2048,280],[],[280],[280],[]],
[[2048,128],[],[128],[128],[]],
[[2048,1320],[],[1320],[1320],[]],
[[2048,360],[],[360],[360],[]],
[[2048,384],[],[384],[384],[]],
[[2048,300],[],[300],[300],[]],
[[2048,1976],[],[1976],[1976],[]],
[[2048,1992],[],[1992],[1992],[]],
[[2048,1980],[],[1980],[1980],[]],
[[2048,1964],[],[1964],[1964],[]],
[[2048,1968],[],[1968],[1968],[]],
[[2048,1888],[],[1888],[1888],[]],
[[2048,1824],[],[1824],[1824],[]],
[[1792,156],[],[156],[156],[]],
[[1792,312],[],[312],[312],[]],
[[1792,736],[],[736],[736],[]],
[[1792,56672],[],[56672],[56672],[]],
[[1792,3072],[],[3072],[3072],[]],
[[1792,5972],[],[5972],[5972],[]],
[[1792,2048],[],[2048],[2048],[]],
[[1792,256],[],[256],[256],[]],
[[1792,61,156],[],[156],[156],[]],
[[1792,3488],[],[3488],[3488],[]],
[[1792,512],[],[512],[512],[]],
[[3072,888],[],[888],[888],[]],
[[3072,160],[],[160],[160],[]],
[[3072,316],[],[316],[316],[]],
[[3072,24],[],[24],[24],[]],
[[3072,128],[],[128],[128],[]],
[[3072,440],[],[440],[440],[]],
[[3072,768],[],[768],[768],[]],
[[3072,256],[],[256],[256],[]],
[[3072,48760],[],[48760],[48760],[]],
[[3072,1536],[],[1536],[1536],[]],
[[3072,4480],[],[4480],[4480],[]],
[[3072,2048],[],[2048],[2048],[]],
[[3072,1024],[],[1024],[1024],[]],
[[3072,57,160],[],[160],[160],[]],
[[3072,121,160],[],[160],[160],[]],
[[3072,4200],[],[4200],[4200],[]],
[[3072,32,160],[],[160],[160],[]],
[[3072,512],[],[512],[512],[]],
[[2048,512],[],[512],[512],[]],

)
for s in shapes:
     run_bw(s)




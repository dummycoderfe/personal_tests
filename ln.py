import torch
import torch.nn as nn
import apex
from torch.profiler import profile, record_function, ProfilerActivity
from apex.normalization import FusedLayerNorm
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
    

model = LayerNormModel([128]).cuda().to(torch.bfloat16)
model2 = LayerNormModelApex([128]).cuda().to(torch.bfloat16)
model.cuda()
model2.cuda()
input_tensor = torch.randn(2048, 130, 128, requires_grad=True).cuda().to(torch.bfloat16)
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
prof.export_chrome_trace("trace_ln.json")
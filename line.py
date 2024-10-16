import torch 
from torch.profiler import profile, record_function, ProfilerActivity

t0 = torch.randn(4096, 1219, 160, dtype = torch.half, device="cuda:1").cuda()
def forward(t0):
    permute = torch.permute(t0, [0, 2, 1])
    cont = permute.contiguous()

def forward2(t0):
    tranpose = torch.transpose(t0, 1, 2)
    cont = tranpose.contiguous()

for _ in range(2):
    forward(t0)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        forward(t0)
torch.cuda.synchronize()
prof.export_chrome_trace("trace2.json")
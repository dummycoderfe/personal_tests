import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch import nn
# # Create a tensor with the shape [[4096, 64, 160], [4096, 64, 160], []]
# # Note: The empty list [] is not valid for a tensor shape, so we'll skip it
# original = torch.randn(4096, 64, 160).half().cuda()
# print(original.shape, original.dtype, original.device   )
# # Copy the tensorfrom torch.profiler import profile, record_function, ProfilerActivity

# copy2 = torch.empty_like(original) #.clone()
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
#     copy1 = original.clone()
#     copy1 = original.clone()
#     copy1 = original.clone()
#     copy1 = original.clone()
#     copy1 = original.clone()
#     copy2.copy_(original)
#     copy2.copy_(original)
#     copy2.copy_(original)
#     copy2.copy_(original)
#     copy2.copy_(original)
#     copy2.copy_(original)

#     # Verify that the copied tensor is indeed a copy
#     # print(torch.sum(torch.abs(copy2 - copy1)))
ori = torch.randn(4096, 160, 1219, dtype=torch.float16).cuda()
ori1 = ori[:,0:32]
_attr_26 = nn.Parameter(torch.randn(160, 32, dtype=torch.float16)).cuda()
_attr_27 = nn.Parameter(torch.randn(160, dtype=torch.float16)).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    linear_176 = torch.nn.functional.linear(
        input=ori1, weight=_attr_26, bias=_attr_27
    )
prof.export_chrome_trace("trace.json")
print(linear_176)
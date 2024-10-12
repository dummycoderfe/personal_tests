import torch
import torch.nn as nn
import apex
from apex.normalization import FusedLayerNorm
# 定义LayerNorm层
class LayerNormModel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormModel, self).__init__()
        self.layer_norm1 = nn.LayerNorm(num_features)
        self.layer_norm2 = nn.LayerNorm(num_features)

    def forward(self, x):
        return self.layer_norm1(self.layer_norm2(x))

# 创建一个具有64个特征的LayerNorm层
model = LayerNormModel(64)
model.cuda()
# 创建一个具有适当shape的输入张量
# 假设我们有一个batch大小为4096，每个样本有200个特征，每个特征有64个元素

input_tensor = torch.randn(4096, 200, 64).cuda()

# 前向传播
output = model(input_tensor)

# 反向传播（假设我们有一些损失函数和目标）
loss = output.mean()  # 只是一个示例损失函数
loss.backward()  # 计算梯度

# 打印输出以验证
print(loss)  # 应该与输入形状相同

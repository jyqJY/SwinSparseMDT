import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


class SwinTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SwinTClassifier, self).__init__()
        self.swin_model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.swin_model.head = nn.Linear(self.swin_model.head.in_features, num_classes)

    def forward(self, x):
        # 直接调用 swin_model，返回输出（形状应为 [batch, num_classes]）
        outputs = self.swin_model(x)
        attn_weights = None  # 暂时不返回注意力权重
        return outputs, attn_weights
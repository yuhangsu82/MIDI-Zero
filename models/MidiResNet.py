import torch
from torch.nn import functional as F
from torchvision.models import resnet34, resnet18, resnet50
from torch import nn
from torch import Tensor


class MidiResNet(nn.Module):
    def __init__(self, emd_size=128):
        super().__init__()
        self.emd_size = emd_size
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

        self.resnet = resnet34(pretrained=True)
        self.backbone = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            nn.Flatten(),
        )
        self.fc = nn.Linear(1536, self.emd_size)
        self.l2_norm = F.normalize

    def forward(self, x, time_info=None) -> torch.Tensor:
        if time_info is not None:
            x = time_info
        x_out = x.unsqueeze(1)
        x_out = self.conv(x_out)
        x_out = self.backbone(x_out)
        x_out = self.fc(x_out)
        x_out = self.l2_norm(x_out, dim=1)

        return x_out



if __name__ == "__main__":
    model = MidiResNet()
    a = torch.rand(2, 15, 88)
    b = torch.rand(2, 15, 88)
    print(model(a).shape)

from torch import nn
import torchvision
import torch

class ObjectDetectionModel(nn.Module):
    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet_body = nn.Sequential(*(list(resnet.children())[:-2]))
        in_channels = 512
        out_channels = self.grid.det_sz * self.grid.num_ancs
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)

    def forward(self, x):
        x = self.resnet_body(x)
        x = self.conv1(x)
        batch_sz = x.shape[0]
        x = x.reshape(self.grid.get_out_shape(batch_sz))

        # Force anchor scales to be >= 0.
        x[:, :, 2:4, :, :] = torch.exp(x[:, :, 2:4, :, :])

        # Force class values to be probs.
        x[:, :, 4:, :, :] = torch.sigmoid(x[:, :, 4:, :, :])

        return x

    def freeze_body(self):
        for p in self.resnet_body.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

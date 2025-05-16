import torch.nn as nn
from torchvision.models import vgg19
import config
import torch


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):

        # converter foto in_channel igual 1 mandar para 3 channel
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)

        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)
    

def test():
    low_resolution = 192
    with torch.amp.autocast("cuda"):
        x = torch.randn((1, 1, low_resolution, low_resolution))
        y = torch.randn((1, 1, low_resolution, low_resolution))
        vgg = VGGLoss()
        print(vgg(x, y))

if __name__ == '__main__':
    test()
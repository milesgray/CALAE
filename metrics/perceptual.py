
from torchvision import models

class PerceptualLoss:
    def __init__(self, ilayer=17+7, device='cuda:0', verbose=False):
        self.vgg = models.vgg16(pretrained=True).to(device).eval()
        if verbose: print(f"VGG: {self.vgg}")
        for param in self.vgg.parameters():
            param.requires_grad = False
        # assumes [-1, 1] input
        def transf(x):
            return (x + 1) * 0.5
        self.trans = transf
        self.name = ilayer

    def __call__(self, x):
        x = self.trans(x)
        layer = self.vgg.features[:self.name+1]
        x = layer(x)
        return x * 0.1
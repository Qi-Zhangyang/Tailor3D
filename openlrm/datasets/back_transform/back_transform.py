import torch
import torchvision.transforms as transforms

# Add gauusian noise 
class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, img):
        return img + torch.randn(img.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def transform_back_image():
    return transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomRotation((-5, 5)),
        transforms.Pad(padding=10, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.1)
    ])

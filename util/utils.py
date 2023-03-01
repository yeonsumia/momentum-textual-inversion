import torch
from torchvision import transforms
import numpy as np


def tensor_to_numpy(x):
    x = x[0].permute(1, 2, 0)
    x = torch.clamp(x, -1, 1)
    x = (x+1) * 127.5
    x = x.cpu().detach().numpy().astype(np.uint8)
    return x

def numpy_to_tensor(x):
    x = (x / 255 - 0.5) * 2
    x = torch.from_numpy(x).unsqueeze(0).permute(0, 3, 1, 2)
    x = x.cuda().float()
    return x

def tensor_to_pil(x):
    x = torch.clamp(x, -1, 1)
    x = (x+1) * 127.5
    return transforms.ToPILImage()(x.squeeze_(0))
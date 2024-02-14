#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib
def tensor_info(tensor):
    """
    Prints information about a PyTorch tensor including min, max, mean, std, and shape.
    
    Args:
        tensor (torch.Tensor): Input tensor
    """
    # print("\nname:", f"{tensor=}")
    print("\nShape:", tensor.shape)
    print("Datatype:", tensor.dtype)
    print("Device:", tensor.device)
    print("Requires grad:", tensor.requires_grad)
    print("Min value:", tensor.min().item())
    print("Max value:", tensor.max().item())
    print("Mean value:", tensor.mean().item())
    print("Standard deviation:", tensor.std().item())

def show(img):
    img = img - img.min()  # Normalize to [0, max]
    img = img / img.max()  # Normalize to [0, 1]
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()
def show_mask(img):
    img = img - img.min()  # Normalize to [0, max]
    img = img / img.max()  # Normalize to [0, 1]
    
    # Detach tensor from computation graph and move to CPU
    npimg = img.detach().cpu().squeeze().numpy()  # Squeeze is used to remove any singleton dimensions
    
    plt.imshow(npimg, interpolation='nearest', cmap='inferno')
    plt.colorbar()
    plt.show()
def show_with_masks(image, masks, mask_names=None, alpha=0.5):
    """
    Overlay masks on an image.
    
    Args:
    - image: PyTorch tensor of shape (C, H, W)
    - masks: List of PyTorch tensors, each of shape (H, W)
    - mask_names: List of names for each mask for the legend
    - alpha: Transparency level for masks
    """
    # Normalize image
    image = image - image.min()
    image = image / image.max()
    npimg = image.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']  # You can add more colors if needed
    
    for idx, mask in enumerate(masks):
        mask = mask - mask.min()
        mask = mask / mask.max()
        npmask = mask.detach().cpu().numpy()

        # Overlay the binary mask with a color
        # First, create a RGB version of the mask where it's colored
        mask_colored = np.zeros((npmask.shape[0], npmask.shape[1], 3))
        for i in range(3):  # for R, G, B channels
            mask_colored[..., i] = npmask * matplotlib.colors.to_rgb(colors[idx % len(colors)])[i]
        
        plt.imshow(mask_colored, interpolation='nearest', alpha=alpha)
    
    if mask_names:
        patches = [plt.Rectangle((0,0),1,1, color=colors[i % len(colors)]) for i in range(len(masks))]
        plt.legend(patches, mask_names, loc='upper left')
    
    plt.show()
def inverse_sigmoid(x):
    return torch.log(x/(1-x))
def var_generalized(gen_sigma,beta=torch.Tensor([2.0])):
    return 0.5* torch.exp(torch.log(gen_sigma)+torch.lgamma(1.0/torch.abs(beta)) - torch.lgamma(3.0/torch.abs(beta)) )
def var_approx(beta=torch.Tensor([2.0]),strength=1.0):
    """_summary_

    Args:
        beta (_type_, optional): _the skewness parameter of geenrlized gaussian_. Defaults to torch.Tensor([2.0]).

    Returns:
        _type_: a scalar tensor for the beta activation
    """
    return torch.relu(2.0* torch.sigmoid(strength*beta))
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent, seed=0):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(torch.device("cuda:0"))

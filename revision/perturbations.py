"""Threat models, defined on UN-normalized [0,1] images.

The original code added Gaussian noise *after* ImageNet normalization, so the
reported sigma was on the normalized scale and was never documented. Here noise
and occlusion are applied to the raw [0,1] image, the result is clamped back to
[0,1], and normalization happens afterwards inside the model wrapper. This makes
sigma directly interpretable (fraction of full dynamic range) and matches the
standard corruption-robustness protocol.
"""

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def add_gaussian_noise(x: torch.Tensor, sigma: float, generator=None) -> torch.Tensor:
    """Add zero-mean Gaussian noise with std `sigma` to a [0,1] image, then clamp.

    `x` is a batch of images in [0,1]. `sigma` is expressed on the same [0,1]
    scale (e.g. sigma=0.1 is 10% of full dynamic range).
    """
    if sigma == 0:
        return x
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return torch.clamp(x + noise * sigma, 0.0, 1.0)


def apply_occlusion(x: torch.Tensor, block_size: int, generator=None) -> torch.Tensor:
    """Mask a random `block_size` x `block_size` square to black (0) on each image.

    On a [0,1] image, 0 is true black, matching the paper's wording (the original
    set 0 on a normalized tensor, which is mid-gray, not black).
    """
    if block_size == 0:
        return x
    out = x.clone()
    n, _, h, w = out.shape
    max_x = max(1, h - block_size)
    max_y = max(1, w - block_size)
    xs = torch.randint(0, max_x, (n,), generator=generator)
    ys = torch.randint(0, max_y, (n,), generator=generator)
    for i in range(n):
        out[i, :, xs[i]:xs[i] + block_size, ys[i]:ys[i] + block_size] = 0.0
    return out


def apply_threat(x: torch.Tensor, threat: str, level, generator=None) -> torch.Tensor:
    if threat == "noise":
        return add_gaussian_noise(x, float(level), generator)
    if threat == "occlusion":
        return apply_occlusion(x, int(level), generator)
    raise ValueError(f"unknown threat: {threat}")

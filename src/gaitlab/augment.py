"""GEI augmentations for training (applied per image, on the GPU).

  shift  : random horizontal translation of up to `max_shift` px (zero fill)
  erase  : random erasing of one rectangle with probability `p_erase`
  flip   : horizontal flip with probability `p_flip` (mirrors walking direction;
           view θ becomes roughly 180°−θ)
"""
import torch


def augment(x, max_shift=0, p_erase=0.0, erase_frac=(0.05, 0.2), p_flip=0.0):
    B, _, H, W = x.shape
    x = x.clone()
    if p_flip > 0:
        m = torch.rand(B, device=x.device) < p_flip
        x[m] = x[m].flip(-1)
    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (B,))
        for i, s in enumerate(shifts.tolist()):
            if s:
                x[i] = torch.roll(x[i], s, dims=-1)
                if s > 0:
                    x[i, :, :, :s] = 0
                else:
                    x[i, :, :, s:] = 0
    if p_erase > 0:
        for i in range(B):
            if torch.rand(1).item() < p_erase:
                area = torch.empty(1).uniform_(*erase_frac).item() * H * W
                ar = torch.empty(1).uniform_(0.3, 3.3).item()
                h = int(min(H, max(1, round((area * ar) ** 0.5))))
                w = int(min(W, max(1, round((area / ar) ** 0.5))))
                t = torch.randint(0, H - h + 1, (1,)).item()
                l = torch.randint(0, W - w + 1, (1,)).item()
                x[i, :, t:t + h, l:l + w] = 0
    return x


def erase_band(x, p=0.5, min_h=6, max_h=16):
    """Zero one random horizontal band (full width) per image with probability p.
    Band height and position are random (not aligned to the 8 strips), so the
    model cannot key on strip boundaries. Simulates occlusion of a body region."""
    B, _, H, W = x.shape
    x = x.clone()
    for i in range(B):
        if torch.rand(1).item() < p:
            h = torch.randint(min_h, max_h + 1, (1,)).item()
            t = torch.randint(0, H - h + 1, (1,)).item()
            x[i, :, t:t + h, :] = 0
    return x

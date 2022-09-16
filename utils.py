import os
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def xyz2uv_stereographic(xyz: torch.Tensor, normalized=False):
    """
    xyz: tensor of size (B, 3)
    """
    if not normalized:
        xyz = xyz / xyz.norm(dim=-1, keepdim=True)
    x, y, z = torch.split(xyz, 1, dim=-1)
    z = torch.clamp_max(z, 0.99)
    denorm = torch.reciprocal(-z + 1)
    u, v = x * denorm, y * denorm
    return torch.cat([u, v], dim=-1)


def uv2xyz_stereographic(uv: torch.Tensor):
    u, v = torch.split(uv, 1, dim=-1)
    u2v2 = u ** 2 + v ** 2
    x = u * 2 / (u2v2 + 1)
    y = v * 2 / (u2v2 + 1)
    z = (u2v2 - 1) / (u2v2 + 1)
    return torch.cat([x, y, z], dim=-1)


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    pixelpoints = np.stack([i, j, np.ones_like(i)], -1)[..., np.newaxis]
    localpoints = np.linalg.inv(K) @ pixelpoints

    rays_d = (c2w[:3, :3] @ localpoints)[..., 0]
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def get_rays_tensor(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()

    pixelpoints = torch.stack([i, j, torch.ones_like(i)], -1).unsqueeze(-1)
    localpoints = torch.matmul(torch.inverse(K), pixelpoints)

    rays_d = torch.matmul(c2w[:3, :3], localpoints)[..., 0]
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def raw2poses(rot_raw, tran_raw, intrin_raw):
    x = rot_raw[..., 0]
    x = x / torch.norm(x, dim=-1, keepdim=True)
    z = torch.cross(x, rot_raw[..., 1])
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x)
    rot = torch.stack([x, y, z], dim=-1)
    pose = torch.cat([rot, tran_raw[..., None]], dim=-1)
    bottom = torch.tensor([0, 0, 1]).type_as(intrin_raw).reshape(-1, 1, 3).expand(len(intrin_raw), -1, -1)
    intrinsic = torch.cat([intrin_raw, bottom], dim=1)
    return pose, intrinsic


def get_batched_rays_tensor(H, W, Ks, c2ws):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()

    pixelpoints = torch.stack([i, j, torch.ones_like(i)], -1)[None, ..., None]
    localpoints = torch.matmul(torch.inverse(Ks)[:, None, None, ...], pixelpoints)

    rays_d = torch.matmul(c2ws[:, None, None, :3, :3], localpoints)[..., 0]
    rays_o = c2ws[:, None, None, :3, -1].expand(rays_d.shape)
    return torch.stack([rays_o, rays_d], dim=1)


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def smart_load_state_dict(model: nn.Module, state_dict: dict):
    if isinstance(model, nn.DataParallel):
        model = model.module

    if "network_state_dict" in state_dict.keys():
        state_dict = {k[7:]: v for k, v in state_dict["network_state_dict"].items()}
        example_keys = [k for k in state_dict.keys() if ".example_" in k]
        for k in example_keys:
            print(f'smart_load_state_dict::pop keys {example_keys}')
            state_dict.pop(k, None)
    else:
        state_dict = state_dict

    is_texture_map = len([k for k in state_dict.keys() if "map.texture_map" in k]) > 0
    if is_texture_map:
        model.texture_map.promote_texture(mlp2map=False)
        if id(model.texture_map) != id(model.texture_map_coarse) \
                and hasattr(model.texture_map_coarse, "promote_texture"):
            model.texture_map_coarse.promote_texture(mlp2map=False)
    model.load_state_dict(state_dict)


LEFT_EYE = [21, 25, 28, 31, 39, 40]
RIGHT_EYE = [22, 24, 29, 30, 34, 37]
NOSE = [12, 27, 33, 43, 49, 57, 56, 54, 55, 53]
MOUTH = [66, 67, 68, 71, 70, 75, 78, 79]
LEFT_EYEBROW = [14, 10, 8, 13, 17]
RIGHT_EYEBROW = [15, 11, 9, 16, 18]
FACE_OUT = [2, 0, 4, 7, 23, 48, 58, 69, 80, 88, 93, 90, 89,
            3, 1, 5, 6, 26, 44, 65, 74, 82, 87, 92, 94, 91, 95]
FACE_IN = [19, 36, 45, 59, 76, 81, 83,
           20, 35, 52, 64, 77, 85, 84, 86,
           60, 46, 32, 50, 61, 41, 72,
           63, 47, 38, 51, 62, 42, 73]


def get_colors():
    color = np.zeros((96, 3), dtype=np.uint8)
    color[LEFT_EYE] = (52, 152, 219)
    color[RIGHT_EYE] = (41, 128, 185)
    color[NOSE] = (230, 126, 34)
    color[MOUTH] = (192, 57, 43)
    color[LEFT_EYEBROW] = (253, 121, 168)
    color[RIGHT_EYEBROW] = (232, 67, 147)
    color[FACE_OUT] = (108, 92, 231)
    color[FACE_IN] = (46, 204, 113)
    return (color / 255).astype(np.float32)

import torch
import torch.nn as nn
import imageio
import cv2
import torch.nn.functional as torchf
import numpy as np
from utils import get_colors


def smooth_scalar(scalar, iternum):
    """
    Args:
        scalar: size T, ...
    """
    for i in range(iternum):
        scalar = torch.cat([scalar[:1], scalar, scalar[-1:]])
        scalar = (scalar[2:] + 2 * scalar[1:-1] + scalar[:-2]) / 4
    return scalar


class WarpProj(nn.Module):
    def __init__(self, time_len, uv_gt_init, uv_gt_id2t, canonicaldir):
        super().__init__()
        pt3d = torch.cat([uv_gt_init[:, :, :3], torch.ones_like(uv_gt_init[:, :, 3:4])], dim=-1)
        if canonicaldir is None:
            pt_tar = pt3d[:1, :, :3]
        else:
            pt_tar = np.load(canonicaldir).astype(np.float32)
            pt_tar = torch.tensor(pt_tar).to(pt3d.device)[None, ...]
        pt_src = pt3d

        pt_tar_T = pt_tar.permute(0, 2, 1)
        pt_src_T = pt_src.permute(0, 2, 1)
        transform = pt_tar_T @ pt_src @ torch.inverse(pt_src_T @ pt_src)
        transform = transform[uv_gt_id2t]
        self.register_buffer("dummy_trans", transform[0:1])
        # we will not optimize the first frame
        self.register_parameter("transform",
                                nn.Parameter(transform[1:], requires_grad=True))

    def get_transform(self):
        return torch.cat([self.dummy_trans, self.transform])

    def forward(self, pts, t):
        """
        Args:
            pts: ..., 3
        """
        ori_shape = pts.shape
        pts = pts.reshape(-1, 3)
        t = t.reshape(-1).type(torch.long)
        trans = self.get_transform()[t]
        pts = trans @ torch.cat([pts, torch.ones_like(pts[:, :1])], dim=-1)[..., None]
        pts = pts[:, :3, 0] / pts[:, 3:4, 0]
        pts = pts.reshape(*ori_shape)
        return pts


class WarpKptAdvanced(nn.Module):
    def __init__(self, uv_gt_init, uv_gt_id2t,
                 canonicaldir,
                 kptdir,
                 affine_perpoint=False,
                 rbf_perframe=False):
        """
        1. ability to model affine_perpoint  2. radius is defined per-dimension 3. ASAP loss support
        """
        super().__init__()
        # estimate a global transform
        pt3d = torch.cat([uv_gt_init[:, :, :3], torch.ones_like(uv_gt_init[:, :, 3:4])], dim=-1)
        pt_tar = np.load(canonicaldir).astype(np.float32)
        pt_tar = torch.tensor(pt_tar).to(pt3d.device)[None, ...]
        pt_src = pt3d

        pt_tar_T = pt_tar.permute(0, 2, 1)
        pt_src_T = pt_src.permute(0, 2, 1)
        transform = pt_tar_T @ pt_src @ torch.inverse(pt_src_T @ pt_src)

        # select keypoints (x)
        kpt_idx = np.load(kptdir).reshape(-1)
        initial_rbf = 10
        self.is_flame = False
        kpt_idx = torch.tensor(kpt_idx).long().to(pt3d.device)
        kpt3d = pt3d[:, kpt_idx]

        # keypoints warp to semi canonical space (Hx)
        kpt3d_remap = (transform @ kpt3d.transpose(-1, -2)).transpose(-1, -2)
        kpt3d_remap = kpt3d_remap[uv_gt_id2t]
        kpt3d = kpt3d[uv_gt_id2t].clone()

        kpt3d_canonical = pt_tar[:, kpt_idx]

        transform = transform[uv_gt_id2t]
        transform = smooth_scalar(transform, 1)

        self.affine_perpoint = affine_perpoint
        if affine_perpoint:
            kpt3d_affine = torch.zeros([kpt3d_remap.shape[0], kpt3d_remap.shape[1], 3, 3]).float()
            self.register_parameter("kpt3d_affine",
                                    nn.Parameter(kpt3d_affine, requires_grad=True))

        self.register_buffer("kpt3d_canonical",
                             kpt3d_canonical)
        self.kpt3d_original = kpt3d.detach().clone()
        self.register_parameter("kpt3d",
                                nn.Parameter(kpt3d_remap, requires_grad=True))

        if rbf_perframe:
            kpt3d_bias_radius = torch.ones_like(kpt3d_remap[..., :3]) * initial_rbf
        else:
            kpt3d_bias_radius = torch.ones_like(kpt3d_canonical[..., :3]) * initial_rbf
        self.register_parameter("kpt3d_bias_radius",
                                nn.Parameter(kpt3d_bias_radius, requires_grad=True))
        self.register_parameter("transform",
                                nn.Parameter(transform, requires_grad=True))

    def forward(self, pts, t):
        """
        Args:
            pts: ..., 3
        """
        ori_shape = pts.shape
        pts = pts.reshape(-1, 3)
        if len(torch.unique(t)) == 1:
            t = int(t.reshape(-1)[0])
            trans = self.transform[t:t + 1]
            kpt3d = self.kpt3d[t:t + 1]
            affine = self.kpt3d_affine[t:t + 1] if self.affine_perpoint else None
            if len(self.kpt3d_bias_radius) > 1:
                kpt3d_bias_radius = self.kpt3d_bias_radius[t:t + 1]
            else:
                kpt3d_bias_radius = self.kpt3d_bias_radius
        else:
            raise RuntimeError("point-wise t not implemented")
        pts = trans @ torch.cat([pts, torch.ones_like(pts[:, :1])], dim=-1)[..., None]
        pts = pts[:, None, ..., 0]

        # compute the weight
        dist = torch.norm(kpt3d - pts, dim=-1, keepdim=True)[..., 0]
        dist, dist_idx = torch.topk(dist, 32, dim=-1, largest=False, sorted=False)
        first_idx = torch.arange(len(dist))[:, None].type_as(dist_idx)
        dist = dist[..., None] * kpt3d_bias_radius.expand(len(dist), -1, -1)[first_idx, dist_idx]
        weight = torch.exp(- dist ** 2).clamp_min(1e-10)

        if self.affine_perpoint:
            delta_kpt = (affine @ (pts - kpt3d)[..., None])[..., 0] + self.kpt3d_canonical - kpt3d
        else:
            delta_kpt = self.kpt3d_canonical - kpt3d

        delta_kpt_pad = delta_kpt.expand(len(dist), -1, -1)[first_idx, dist_idx]
        delta_pts = (delta_kpt_pad * weight).sum(dim=-2) / weight.sum(dim=-2)
        pts = pts.reshape(*ori_shape) + delta_pts.reshape(*ori_shape)
        return pts

    def get_kpts_world(self):
        # return self.kpt3d_original[..., :3]
        kpt = self.transform[:, :3, :3].inverse()[:, None] @ (self.kpt3d - self.transform[:, None, :3, 3])[..., None]
        return kpt[..., 0]

    def compute_kpt_loss(self, t):
        kpt3d_original = self.kpt3d_original[t].type_as(self.kpt3d)
        trans = self.transform[t]
        kpt3d_remap = kpt3d_original @ trans.T
        return (kpt3d_remap - self.kpt3d[t]).abs().mean()

    def inverse_forward(self, pts, t):
        """
        Args:
            pts: ..., 3
        """
        ori_shape = pts.shape
        pts = pts.reshape(-1, 3)
        trans = self.transform[t:t + 1]
        trans = torch.cat([trans, torch.tensor([0, 0, 0, 1.]).reshape(1, 1, 4).type_as(trans)], dim=-2).inverse()[:, :3]
        kpt3d = self.kpt3d[t:t + 1]
        if len(self.kpt3d_bias_radius) > 1:
            kpt3d_bias_radius = self.kpt3d_bias_radius[t:t + 1]
        else:
            kpt3d_bias_radius = self.kpt3d_bias_radius

        # compute the weight
        dist = torch.norm(self.kpt3d_canonical - pts[:, None], dim=-1, keepdim=True) * kpt3d_bias_radius
        weight = torch.exp(- dist ** 2).clamp_min(1e-6)
        delta_kpt = kpt3d - self.kpt3d_canonical
        delta_pts = (delta_kpt * weight).sum(dim=-2) / weight.sum(dim=-2)
        pts = pts + delta_pts

        pts = trans @ torch.cat([pts, torch.ones_like(pts[:, :1])], dim=-1)[..., None]
        return pts.reshape(ori_shape)

    def render(self, pts, t):
        old_shape = pts.shape
        pts = pts.reshape(-1, 3)
        kpts = self.get_kpts_world()[int(t.reshape(-1)[0])]
        dist = pts[:, None] - kpts[None, :]
        dist = torch.norm(dist, dim=-1)
        sigma = (torch.exp(-(dist ** 2) * 30000) * 50000).sum(dim=-1)
        pt_color = torch.tensor(get_colors())
        color_idx = torch.argmin(dist, dim=-1)
        color = pt_color[color_idx]
        return sigma.reshape(old_shape[:-1]), color.reshape(old_shape).clamp(0.0001, 0.9999)

    def deform(self, new_pts, t):
        trans = self.transform[t]
        new_pts = torch.cat([new_pts, torch.ones_like(new_pts[:, :1])], dim=-1)
        new_pts_deformed = new_pts @ trans.T
        pts_delta = new_pts_deformed - self.kpt3d[t]
        self.kpt3d += pts_delta[None, ...]
        return

    def stable2first(self):
        with torch.no_grad():
            self.transform.data[:] = self.transform[:1]
            if self.affine_perpoint:
                self.kpt3d_affine.data[:] = self.kpt3d_affine[:1]
            self.kpt3d.data[:] = self.kpt3d[:1]
            self.kpt3d_bias_radius.data[:] = self.kpt3d_bias_radius[:1]

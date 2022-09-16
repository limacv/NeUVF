import os
import torch
import imageio
import numpy as np
import math
import torch.nn as nn
import time
from NeRF import *

from configs import config_parser
from dataloader import load_data, load_images, load_masks, load_position_maps, has_matted, load_matted
from utils import *
import shutil
from datetime import datetime
from metrics import compute_img_metric
import cv2
from PIL import Image

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_texture_map(nerf: NeUVFModulateT, resolution, ts, poses):
    poses = torch.tensor(poses)
    poses_dir = poses[:, :3, :3] * torch.tensor([0, 0, 1.]).type_as(poses)[None, None, :]
    poses_dir = poses_dir.sum(dim=-1)
    viewdir_mid = poses_dir[4]
    viewdir_up = poses_dir[3]
    viewdir_down = poses_dir[8] + poses_dir[11]
    viewdir_left = poses_dir[6] + torch.tensor([2, 0, -0.5])
    viewdir_right = poses_dir[2] + torch.tensor([-2, 0, -0.5])
    viewdir_up = (viewdir_up / viewdir_up.norm())[None, None, :]
    viewdir_down = (viewdir_down / viewdir_down.norm())[None, None, :]
    viewdir_left = (viewdir_left / viewdir_left.norm())[None, None, :]
    viewdir_right = (viewdir_right / viewdir_right.norm())[None, None, :]

    y, x = torch.meshgrid([torch.linspace(0, 1, resolution), torch.linspace(0, 1, resolution)])
    x, y = x[..., None], y[..., None]
    viewdir = viewdir_up * (1 - y) + viewdir_down * y + viewdir_left * (1 - x) + viewdir_right * x
    viewdir = viewdir / viewdir.norm(dim=-1, keepdim=True)
    baked_list = []
    residual_list = []
    albedo_list = []
    for ti in ts:
        textures = nerf.get_texture_map(resolution, ti, views=viewdir)
        baked, albedo, residual = textures[-1].permute(0, 2, 3, 1)[0], textures[1].permute(0, 2, 3, 1)[0], textures[0].permute(0, 2, 3, 1)[0]
        baked_list.append(baked.cpu())
        albedo_list.append(albedo.cpu())
        residual_list.append(residual.cpu())
    return baked_list, albedo_list, residual_list


if __name__ == "__main__":
    parser = config_parser()
    parser.add_argument("--texresolution", type=int, default=1024,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--t", type=str, default='-1',
                        help='#, or #,# or -1')

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    imgpaths, poses, intrinsics, bds, render_poses, render_intrinsics = load_data(datadir=args.datadir,
                                                                                  factor=args.factor,
                                                                                  bd_factor=args.bd_factor,
                                                                                  frm_num=args.frm_num)
    T = len(imgpaths)
    V = len(imgpaths[0])
    H, W = imageio.imread(imgpaths[0][0]).shape[0:2]
    print('Loaded llff', T, V, H, W, poses.shape, intrinsics.shape, render_poses.shape, render_intrinsics.shape,
          bds.shape)
    args.time_len = T
    #######
    # load uv map
    uv_gts = None
    basenames = [os.path.basename(ps_[0]).split('.')[0] for ps_ in imgpaths]
    period = args.uv_map_gt_skip_num + 1
    basenames = basenames[::period]
    uv_gt_id2t = np.arange(0, T, period)
    assert (len(uv_gt_id2t) == len(basenames))
    t2uv_gt_id = np.repeat(np.arange(len(basenames)), period)[:T]
    uv_gts = load_position_maps(args.datadir, args.factor, basenames)
    uv_gts = torch.tensor(uv_gts).cuda()
    # transform uv from (0, 1) to (- uv_map_face_roi,  uv_map_face_roi)
    uv_gts[..., 3:] = uv_gts[..., 3:] * (2 * args.uv_map_face_roi) - args.uv_map_face_roi

    args.uv_gts = uv_gts
    args.t2uv_gt_id = t2uv_gt_id
    nerf = NeUVFModulateT(args)
    ##########################
    # Load checkpoints
    ckpts = [os.path.join(args.expdir, args.expname, f)
             for f in sorted(os.listdir(os.path.join(args.expdir, args.expname))) if 'tar' in f]
    print('Found ckpts', ckpts)

    start = 0
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        smart_load_state_dict(nerf, ckpt)
        if 'rot_raw' in ckpt.keys():
            print("Loading poses and intrinsics from the ckpt")
            rot_raw = ckpt['rot_raw']
            tran_raw = ckpt['tran_raw']
            intrin_raw = ckpt['intrin_raw']
            poses, intrinsics = raw2poses(
                torch.cat([rot_raw0, rot_raw]),
                torch.cat([tran_raw0, tran_raw]),
                torch.cat([intrin_raw0, intrin_raw]))
            assert len(rot_raw) + 1 == V
    render_kwargs_train = {
        'N_samples': args.N_samples,
        'N_importance': args.N_importance,
        'use_viewdirs': args.use_viewdirs,
        'perturb': args.perturb,
        'raw_noise_std': args.raw_noise_std,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    bds_dict = {
        'box': bds
    }
    print(bds_dict)

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    global_step = start

    # ##################################################################################################
    print("Scripting::Finish loading everything!!!=========================================")
    savedir = os.path.join(args.expdir, args.expname, f'texture')
    os.makedirs(savedir, exist_ok=True)

    prefix = ''
    ts = np.arange(T)
    if args.t != '-1':
        if ',' in args.t and ':' not in args.t:
            time_range = list(map(int, args.t.split(',')))
            ts = ts[time_range]
        elif ':' in args.t:
            slices = args.t.split(',')
            ts = []
            for slic in slices:
                start, end = list(map(int, slic.split(':')))
                step = 1 if start <= end else -1
                ts.append(np.arange(start, end, step))
            ts = np.concatenate(ts)
        else:
            time_range = [int(args.t)]
            ts = ts[time_range]

        prefix += args.t.replace(',', '_').replace(':', 't')

    save_image = len(ts) < 20

    print(f"Scripting::saving textures of time {ts}")
    if len(args.texture_map_post) > 0:
        assert os.path.isfile(args.texture_map_post)
        print(f'loading texture map from {args.texture_map_post}')
        assert not args.texture_map_force_map
        nerf.force_load_texture_map(args.texture_map_post, args.texture_map_post_isfull, False)
        prefix += os.path.basename(args.texture_map_post).split('.')[0] + '_'

    with torch.no_grad():
        bakeds, albedos, residuals = export_texture_map(nerf, args.texresolution, ts, poses)

    albedo_save, baked_save, residuals_save = [], [], []
    for t, baked, albedo, residual in zip(ts, bakeds, albedos, residuals):
        base_name = basenames[t]
        baked = np.clip(baked.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        albedo = np.clip(albedo.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        residual = np.clip(residual.cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # draw keypoint overlay
        iskeypoint = ''
        if args.render_keypoints:
            iskeypoint = 'kpt'
            from utils import get_colors
            kpts = nerf.explicit_warp.get_kpts_world()[t]
            viewdirs = torch.randn_like(kpts)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            pts_viewdir = torch.cat([kpts, viewdirs], dim=-1)
            uvs, _, _, _ = nerf(H, W, t=t, chunk=args.chunk, pts_viewdir=pts_viewdir, **render_kwargs_train)
            uvs = (uvs + 1) / 2
            colors = (get_colors() * 255).astype(np.uint8)
            for uv, color in zip(uvs, colors):
                x, y = int(uv[0] * args.texresolution), int(uv[1] * args.texresolution)
                cv2.circle(baked, (x, y), 7, (int(color[0]), int(color[1]), int(color[2])), -1)

        if save_image:
            imageio.imwrite(f"{savedir}/{prefix}albedo_{base_name}.png", albedo)
            imageio.imwrite(f"{savedir}/{prefix}baked_{base_name}{iskeypoint}.png", baked)
            imageio.imwrite(f"{savedir}/{prefix}residual_{base_name}.png", residual)
        else:
            albedo_save.append(albedo)
            baked_save.append(baked)
            residuals_save.append(residual)

    if not save_image:
        imageio.mimwrite(f"{savedir}/{prefix}albedos.mp4", albedo_save, fps=30, quality=8)
        imageio.mimwrite(f"{savedir}/{prefix}bakeds.mp4", baked_save, fps=30, quality=8)
        imageio.mimwrite(f"{savedir}/{prefix}residuals.mp4", residuals_save, fps=30, quality=8)

    print(f"Successfully save textures to {savedir}")


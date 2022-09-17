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

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = config_parser()
    parser.add_argument("--t", type=str, default="-1",
                        help='render time range')
    parser.add_argument("--render_factor", type=float, default=1,
                        help='render the resolution <factor> times of the training resolution')
    parser.add_argument("--texture_map_post", type=str, default='',
                        help='only use when render_only is true, manually specify the map texture')
    parser.add_argument("--texture_map_post_isfull", action='store_true', default=True,
                        help='load texture map as the entire texture instead of the texture roi')
    parser.add_argument("--texture_map_force_map", action='store_true',
                        help='only render the map without the MLP residual')
    parser.add_argument("--render_view", type=int, default=-1,
                        help='render the view index specified by the training data')
    parser.add_argument("--render_deformed", type=str, default='',
                        help='specify the edited file saved from the UI')
    parser.add_argument("--use_deform_pose", action='store_true',
                        help='if true, use the camera pose in the deformation file')
    parser.add_argument("--render_depth", action='store_true',
                        help='if true save depth map as npy')
    parser.add_argument("--render_stable", action='store_true',
                        help='if true render the first frame')

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
    args.roibox = bds

    #######
    # load uv map
    uv_gts = None
    basenames = [os.path.basename(ps_[0]).split('.')[0] for ps_ in imgpaths]
    period = args.uv_map_gt_skip_num + 1
    basenames = basenames[::period]
    uv_gt_id2t = np.arange(0, T, period)
    assert(len(uv_gt_id2t) == len(basenames))
    t2uv_gt_id = np.repeat(np.arange(len(basenames)), period)[:T]
    print("load position maps")
    args.uv_gts = torch.rand(T, 36942, 5)
    args.t2uv_gt_id = np.arange(T)

    if args.nerf_type == 'NeRFModulateT':
        nerf = NeRFModulateT(args)
    elif args.nerf_type == 'NeUVFModulateT':
        nerf = NeUVFModulateT(args)
    elif args.nerf_type == 'NeRFTemporal':
        nerf = NeRFTemporal(args)
    else:
        raise RuntimeError(f"nerf_type {args.nerf_type} not recognized")

    nerf = nn.DataParallel(nerf, [0, ])

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

    global_step = start
    suffix = ''

    # ##################################################################################################
    print("Scripting::Finish loading everything!!!")
    # deciding render view and range
    poses = torch.tensor(poses)
    intrinsics = torch.tensor(intrinsics)
    render_poses = torch.tensor(render_poses).to(device)
    render_intrinsics = torch.tensor(render_intrinsics).to(device)

    render_t = np.arange(T)
    if args.render_view >= 0:
        render_poses = poses[args.render_view:args.render_view + 1].expand(T, -1, -1)
        render_intrinsics = intrinsics[args.render_view:args.render_view + 1].expand(T, -1, -1)
        suffix += f"_view{args.render_view:03d}"

    if args.t != '-1':  # parse the t
        if ',' in args.t and ':' not in args.t:
            time_range = list(map(int, args.t.split(',')))
            render_t = render_t[time_range]
        elif ':' in args.t:
            slices = args.t.split(',')
            render_t = []
            for slic in slices:
                start, end = list(map(int, slic.split(':')))
                step = 1 if start <= end else -1
                render_t.append(np.arange(start, end, step))
            render_t = np.concatenate(render_t)
        else:
            time_range = [int(args.t)]
            render_t = render_t[time_range]

    if len(render_t) > 1:
        pose_i = (np.arange(len(render_t)) / (len(render_t) - 1) * (len(render_poses) - 1)).astype(np.int64)
    else:
        pose_i = [0]
    render_poses = render_poses[pose_i]
    render_intrinsics = render_intrinsics[pose_i]

    print(pose_i)
    print(f'RENDER ONLY ==============================================\n'
          f'Render time {render_t}')
    with torch.no_grad():

        i = start + 1
        if hasattr(nerf.module, "update_step"):
            nerf.module.update_step(i)
        if len(args.texture_map_post) > 0:
            assert os.path.isfile(args.texture_map_post)
            print(f'loading texture map from {args.texture_map_post}')
            nerf.module.force_load_texture_map(args.texture_map_post, args.texture_map_post_isfull, args.texture_map_force_map)
            suffix += '_' + os.path.basename(args.texture_map_post).split('.')[0]
            if args.texture_map_force_map:
                suffix += "force"

        savedir = os.path.join(args.expdir, args.expname, f'render_only_images')

        if args.render_keypoints:
            suffix += "_kpts"

        if len(args.render_deformed) > 0:
            assert hasattr(nerf.module, "explicit_warp") and hasattr(nerf.module.explicit_warp, "kpt3d") \
                   and hasattr(nerf.module.explicit_warp, "transform")

            suffix += os.path.basename(args.render_deformed).split('.')[0]

            new_kpts = np.load(args.render_deformed)

            explicit_warp: WarpKptAdvanced = nerf.module.explicit_warp
            new_cpts = torch.tensor(new_kpts['cpts'])
            explicit_warp.deform(new_cpts, new_kpts['frameidx'])
            if args.use_deform_pose and 'pose' in new_kpts.keys():
                print(f"load camera poses from {args.render_deformed}")
                pose = new_kpts['pose']
                intrin = new_kpts['intrin']
                intrin[0] *= W
                intrin[1] *= W
                new_render_poses = torch.tensor(pose)[None, :3, :].expand_as(render_poses)
                new_render_intrinsics = torch.tensor(intrin)[None, ...].expand_as(render_intrinsics)
                render_poses, render_intrinsics = new_render_poses, new_render_intrinsics

        if args.render_canonical:
            assert hasattr(nerf.module, "explicit_warp")
            print("Rendering canonical, setting explicit_warp to None")
            nerf.module.explicit_warp = None
            suffix += "_canonical"

        if args.render_stable:
            assert hasattr(nerf.module, "explicit_warp")
            print("Rendering canonical, setting explicit_warp to None")
            nerf.module.explicit_warp.stable2first()
            suffix += "_stable"

        if len(suffix) == 0:
            suffix = "original"
        savedir = os.path.join(savedir, suffix)
        os.makedirs(savedir, exist_ok=True)
        with torch.no_grad():
            nerf.eval()
            for ti, (render_time, rpose, rintrin) in enumerate(zip(render_t, render_poses, render_intrinsics)):
                rH, rW = int(H * args.render_factor), int(W * args.render_factor)

                rintrin = rintrin.clone()
                rintrin[:2, :3] *= args.render_factor
                rgbs, disps = nerf(rH, rW, chunk=args.render_chunk, t=render_time,
                                   poses=rpose[None, ...],
                                   intrinsics=rintrin[None, ...])
                rgbs = rgbs[0]
                disps = disps[0]
                rgbs = rgbs.cpu().numpy()
                disps = disps.cpu().numpy()
                basename = basenames[render_time]
                imageio.imwrite(os.path.join(
                    savedir, f"{args.expname}_f{ti:04d}_{basename}.png"
                ), to8b(rgbs))
                if args.render_depth:
                    np.save(os.path.join(
                        savedir, f"{args.expname}_f{ti:04d}_{basename}_depth.npy"
                    ), disps)

        if len(render_t) > 90:
            print("generating video")
            import glob, imageio

            def add_bg(img):
                if img.shape[-1] == 3:
                    return img
                elif img.shape[-1] == 4:
                    img = img.astype(np.float32)
                    alpha = img[..., 3:] / 255
                    img = img[..., :3] * alpha + 28 * (1 - alpha)
                    return np.clip(img, 0, 255).astype(np.uint8)

            images = imageio.mimwrite(os.path.join(savedir, "avideo.mp4"),
                                      [add_bg(imageio.imread(p)) for p in sorted(glob.glob(f"{savedir}/*.png"))],
                                      fps=30,
                                      quality=8)

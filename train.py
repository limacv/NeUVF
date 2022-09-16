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


def train():
    parser = config_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # set up multi-processing
    if args.gpu_num == -1:
        args.gpu_num = torch.cuda.device_count()
        print(f"Using {args.gpu_num} GPU(s)")

    imgpaths, poses, intrinsics, bds, render_poses, render_intrinsics = load_data(datadir=args.datadir,
                                                                                  factor=args.factor,
                                                                                  bd_factor=args.bd_factor,
                                                                                  frm_num=args.frm_num,
                                                                                  frm_start=args.frm_start)
    has_matted_image = has_matted(imgpaths[0])
    if has_matted_image:
        print('Has matted image, load rgba from images_rgba')
    else:
        print('Couldn\'t find matted image, will use images and masks')

    T = len(imgpaths)
    V = len(imgpaths[0])
    H, W = imageio.imread(imgpaths[0][0]).shape[0:2]
    print('Loaded llff', T, V, H, W, poses.shape, intrinsics.shape, render_poses.shape, render_intrinsics.shape,
          bds.shape)
    args.time_len = T
    args.roibox = bds

    # Move testing data to GPU
    render_poses = torch.tensor(render_poses).to(device)
    render_intrinsics = torch.tensor(render_intrinsics).to(device)

    test_ids = [int(id_) for id_ in args.test_ids.split(',')]
    train_ids = np.array([i for i in np.arange(int(V)) if i not in test_ids]).tolist()
    print('Training views are', train_ids)
    print('Test views are', test_ids)

    #######
    # load uv map
    basenames = [os.path.basename(ps_[0]).split('.')[0] for ps_ in imgpaths]
    period = args.uv_map_gt_skip_num + 1
    basenames = basenames[::period]
    uv_gt_id2t = np.arange(0, T, period)
    assert(len(uv_gt_id2t) == len(basenames))
    t2uv_gt_id = np.repeat(np.arange(len(basenames)), period)[:T]
    print("load position maps")
    uv_gts = load_position_maps(args.datadir, args.factor, basenames)
    uv_gts = torch.tensor(uv_gts).cuda()
    # transform uv from (0, 1) to (- uv_map_face_roi,  uv_map_face_roi)
    uv_gts[..., 3:] = uv_gts[..., 3:] * (2 * args.uv_map_face_roi) - args.uv_map_face_roi

    args.uv_gts = uv_gts
    args.t2uv_gt_id = t2uv_gt_id

    # Summary writers
    writer = SummaryWriter(os.path.join(args.expdir, args.expname))

    # Create log dir and copy the config file
    file_path = os.path.join(args.expdir, args.expname, f"source_{datetime.now().timestamp():.0f}")
    os.makedirs(file_path, exist_ok=True)
    if args.config is not None:
        f = os.path.join(file_path, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    files_copy = ['NeRF.py', 'NeRF_modules.py', 'train.py', 'WARP_modules.py',
                  'utils.py', 'dataloader.py', 'configs.py', 'metrics.py']
    for fc in files_copy:
        shutil.copyfile(f"./{fc}", os.path.join(file_path, fc))

    # Create nerf model
    if args.nerf_type == 'NeRFModulateT':
        nerf = NeRFModulateT(args)
    elif args.nerf_type == 'NeUVFModulateT':
        nerf = NeUVFModulateT(args)
    elif args.nerf_type == 'NeRFTemporal':
        nerf = NeRFTemporal(args)
    else:
        raise RuntimeError(f"nerf_type {args.nerf_type} not recognized")

    nerf = nn.DataParallel(nerf, list(range(args.gpu_num)))
    optimizer = torch.optim.Adam(params=nerf.parameters(), lr=args.lrate, betas=(0.9, 0.999))

    ######################
    # if optimize poses
    poses = torch.tensor(poses)
    intrinsics = torch.tensor(intrinsics)
    if args.optimize_poses:
        rot_raw = poses[:, :3, :2]
        tran_raw = poses[:, :3, 3]
        intrin_raw = intrinsics[:, :2, :3]

        # leave the first pose unoptimized
        rot_raw0, tran_raw0, intrin_raw0 = rot_raw[:1], tran_raw[:1], intrin_raw[:1]
        rot_raw = nn.Parameter(rot_raw[1:], requires_grad=True)
        tran_raw = nn.Parameter(tran_raw[1:], requires_grad=True)
        intrin_raw = nn.Parameter(intrin_raw[1:], requires_grad=True)
        pose_optimizer = torch.optim.SGD(params=[rot_raw, tran_raw, intrin_raw],
                                         lr=args.lrate / 5)
    else:
        rot_raw0, tran_raw0, intrin_raw0 = None, None, None
        rot_raw, tran_raw, intrin_raw = None, None, None
        pose_optimizer = None

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
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
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

    print('Begin', args.batch_size)
    global_step = start
    start = start + 1
    N_iters = args.N_iters + 1
    batch_size = args.batch_size

    i_batch0, i_batch1 = 0, 0
    batch_size1 = int(batch_size * args.masked_sample_precent + 0.5)
    batch_size0 = batch_size - batch_size1
    t_list = []
    if args.best_frame_idx >= 0:
        t_list = [args.best_frame_idx] * int(T * 1.5)
    ini_iteration_count = len(t_list) * args.itertions_per_frm
    t_loaded = -1
    per_frm_batch_size = H * W * len(train_ids)

    rays = get_batched_rays_tensor(H, W, intrinsics, poses)
    rays_raw = rays[train_ids].permute(0, 2, 3, 1, 4).reshape(-1, 2, 3).detach()
    for i in range(start, N_iters):
        # if i_batch >= per_frm_batch_size or i == start:
        if i_batch0 + i_batch1 >= per_frm_batch_size or i == start:

            # update training time
            if len(t_list) == 0:
                t_list = np.arange(T)
                np.random.shuffle(t_list)
            t = t_list[0]
            t_list = np.delete(t_list, 0)

            # maybe load the rays
            if t != t_loaded:  # need update:
                if has_matted_image:
                    rgba = load_matted(imgpaths[t])[:, None, ...]
                    msks = (rgba[..., 3:4] > (1. / 255)).astype(np.float32)
                else:
                    rgbs = load_images(imgpaths[t])[:, None, ...]
                    msks = load_masks(imgpaths[t])[:, None, ..., None]
                    rgbs = rgbs * msks
                    rgba = np.concatenate([rgbs, msks], axis=-1)
                msks = msks.reshape(V, H, W)[train_ids]
                msks = torch.tensor(msks).to(device)
                msks_raw = (msks * 255).type(torch.uint8)
                rgba = torch.tensor(rgba, dtype=torch.float32)
                rgba_raw = rgba[train_ids].permute(0, 2, 3, 1, 4).reshape(-1, 4)
                t_loaded = t

            rand_idx = torch.randperm(rgba_raw.shape[0])
            is_mask1 = msks_raw.reshape(-1)[rand_idx]  # select masked idx and unmasked idx
            rand_idx_ma1 = rand_idx[is_mask1 > 250]  # 1 is the foreground and 0 is the background
            rand_idx_ma0 = rand_idx[is_mask1 < 5]
            per_frm_batch_size = min(len(rand_idx), batch_size * args.itertions_per_frm)
            i_batch0, i_batch1 = 0, 0

        # if smooth loss, then remove half of the batch and sample near the batch
        if args.smooth_loss_weight > 0:
            batch_idx = torch.cat([
                rand_idx_ma0[i_batch0: i_batch0 + batch_size0: 2],
                rand_idx_ma1[i_batch1: i_batch1 + batch_size1: 2]
            ])
            batch_idx = batch_idx[:len(batch_idx) // 2]
            batch_W = batch_idx % W
            batch_H = batch_idx // W % H
            batch_V = batch_idx // (H * W)
            ishorizontal = torch.randint(0, 2, batch_H.shape).bool()
            batch_W[ishorizontal] += 1
            batch_W[batch_W >= W] -= 2
            isvertical = torch.logical_not(ishorizontal)
            batch_H[isvertical] += 1
            batch_H[batch_H >= H] -= 2
            batch_idx1 = batch_V * (H * W) + batch_H * W + batch_W
            batch_idx = torch.stack([batch_idx, batch_idx1], dim=1)
            batch_idx = batch_idx.reshape(-1)
        else:  # normal sampling
            batch_idx = torch.cat([
                rand_idx_ma0[i_batch0: i_batch0 + batch_size0],
                rand_idx_ma1[i_batch1: i_batch1 + batch_size1]
            ])

        # if optimizing camera poses, regenerating the rays
        if args.optimize_poses and global_step >= args.optimize_poses_start:
            poses, intrinsics = raw2poses(
                torch.cat([rot_raw0, rot_raw]),
                torch.cat([tran_raw0, tran_raw]),
                torch.cat([intrin_raw0, intrin_raw]))
            rays = get_batched_rays_tensor(H, W, intrinsics, poses)
            rays_raw = rays[train_ids].permute(0, 2, 3, 1, 4).reshape(-1, 2, 3)

        batch_rays = rays_raw[batch_idx]
        target_s = rgba_raw[batch_idx]
        i_batch0 += batch_size0
        i_batch1 += batch_size1

        #####  Core optimization loop  #####
        nerf.train()
        if hasattr(nerf.module, "update_step"):
            nerf.module.update_step(global_step)
        if hasattr(nerf.module, "set_explicit_warp_grad"):
            grad_explicit = global_step >= ini_iteration_count
            nerf.module.set_explicit_warp_grad(grad_explicit)
        rgba, rgba0, extra = nerf(H, W, t=t, rays=batch_rays.reshape(-1, 2, 3), chunk=args.chunk)

        # RGB loss
        img_loss = img2mse(rgba, target_s)
        psnr = mse2psnr(img_loss)

        if args.not_supervise_rgb0 or rgba0 is None:
            print("Warning!! not supervising rgb0")
            img_loss0 = 0
        else:
            img_loss0 = img2mse(rgba0, target_s)

        # UV loss
        if args.uv_loss_weight > 0:
            uv_id = t2uv_gt_id[t]
            uv_t = uv_gt_id2t[uv_id]
            uv_gt = uv_gts[uv_id]
            if args.uv_batch_size > len(uv_gt):
                uv_gt_batch = uv_gt
            else:
                selection = np.random.choice(len(uv_gt), args.uv_batch_size, replace=False)
                uv_gt_batch = uv_gt[selection]
            pts, uv_target = uv_gt_batch.split([3, 2], dim=-1)

            if args.uv_loss_noise_std > 0:
                noise = torch.randn_like(pts) * args.uv_loss_noise_std
                pts += noise
            viewdirs = torch.randn_like(pts)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            pts_viewdir = torch.cat([pts, viewdirs], dim=-1)
            uv, uv0, _, _ = nerf(H, W, t=uv_t, chunk=args.chunk, pts_viewdir=pts_viewdir)

            uv_loss0 = torch.norm(uv0 - uv_target, dim=-1).mean()
            uv_loss = torch.norm(uv - uv_target, dim=-1).mean()
            uv_loss = uv_loss0 + uv_loss

            uv_decay_steps = args.uv_loss_decay * 1000
            new_uv_loss_weight = args.uv_loss_weight * (0.1 ** (global_step / uv_decay_steps))
            if new_uv_loss_weight < 0.002:
                print("Removing the uv smooth term")
                args.uv_loss_weight = 0
        else:
            uv_loss = 0
            new_uv_loss_weight = 0

        # Sparsity loss
        if args.sparsity_loss_weight > 0:
            new_sparisty_loss_weight = args.sparsity_loss_weight \
                if global_step > args.sparsity_loss_start_step else 0
            sparsity_loss = extra["sparsity"].mean()
        else:
            sparsity_loss = 0
            new_sparisty_loss_weight = 0

        # neutex_cycle_loss
        if args.cycle_loss_weight > 0:
            cycle_decay_steps = args.cycle_loss_decay * 1000
            new_cycle_loss_weight = args.cycle_loss_weight * (0.1 ** (global_step / cycle_decay_steps))
            if new_cycle_loss_weight < 0.002:
                print("Removing the cycle loss term")
                args.cycle_loss_weight = 0
            cycle_loss = extra["cycle"].mean()
        else:
            new_cycle_loss_weight = 0
            cycle_loss = 0

        # alpha sparisty loss
        if args.alpha_loss_weight > 0:
            alpha_decay_steps = args.alpha_loss_decay * 1000
            new_alpha_loss_weight = args.alpha_loss_weight * (0.1 ** (global_step / alpha_decay_steps))
            alpha_loss = extra["alpha"].mean()
        else:
            new_alpha_loss_weight = 0
            alpha_loss = 0

        if args.smooth_loss_weight > 0:
            smooth_loss = extra["smooth"].mean()
            new_smooth_loss_weight = args.smooth_loss_weight * min(1, global_step / args.smooth_loss_start_decay)
        else:
            smooth_loss = 0
            new_smooth_loss_weight = 0

        if args.temporal_loss_weight > 0:
            temporal_loss = extra["temporal"].mean()
            new_temporal_loss_weight = args.temporal_loss_weight \
                if global_step > args.temporal_loss_start_step else 0
        else:
            temporal_loss = 0
            new_temporal_loss_weight = 0

        if args.dsmooth_loss_weight > 0:
            dsmooth_loss = extra["d_smooth"].mean()
            new_dsmooth_loss_weight = args.dsmooth_loss_weight
        else:
            dsmooth_loss = 0
            new_dsmooth_loss_weight = 0

        if args.uvsmooth_loss_weight > 0:
            uvsmooth_loss = extra["uv_smooth"].mean()
            new_uvsmooth_loss_weight = args.uvsmooth_loss_weight
        else:
            uvsmooth_loss = 0
            new_uvsmooth_loss_weight = 0

        if args.uvprepsmooth_loss_weight > 0:
            uvprepsmooth_loss = extra["uvp_smooth"].mean()
            new_uvprepsmooth_loss_weight = args.uvprepsmooth_loss_weight
        else:
            uvprepsmooth_loss = 0
            new_uvprepsmooth_loss_weight = 0

        if args.gsmooth_loss_weight > 0:
            gsmooth_decay_steps = args.gsmooth_loss_decay * 1000
            gsmooth_loss = extra["g_smooth"].mean()
            new_gsmooth_loss_weight = args.gsmooth_loss_weight * (0.1 ** (global_step / gsmooth_decay_steps))
        else:
            gsmooth_loss = 0
            new_gsmooth_loss_weight = 0

        if args.kpt_loss_weight > 0:
            kpt_loss = extra["kpt"].mean()
            new_kpt_loss_weight = args.kpt_loss_weight
        else:
            kpt_loss = 0
            new_kpt_loss_weight = 0

        loss = img_loss + img_loss0 \
               + new_uv_loss_weight * uv_loss \
               + new_sparisty_loss_weight * sparsity_loss \
               + new_cycle_loss_weight * cycle_loss \
               + new_alpha_loss_weight * alpha_loss \
               + new_smooth_loss_weight * smooth_loss \
               + new_temporal_loss_weight * temporal_loss \
               + new_dsmooth_loss_weight * dsmooth_loss \
               + new_uvsmooth_loss_weight * uvsmooth_loss \
               + new_uvprepsmooth_loss_weight * uvprepsmooth_loss \
               + new_gsmooth_loss_weight * gsmooth_loss \
               + new_kpt_loss_weight * kpt_loss

        optimizer.zero_grad()
        if args.optimize_poses and global_step >= args.optimize_poses_start:
            pose_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.optimize_poses and global_step >= args.optimize_poses_start:
            pose_optimizer.step()
        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        ################################

        if i % args.i_tensorboard == 0:
            writer.add_scalar('aloss/psnr', psnr, i)
            writer.add_scalar('aloss/mse_loss', loss, i)
            writer.add_scalar('otherloss/uv_loss', uv_loss, i)
            writer.add_scalar('otherloss/alpha_loss', alpha_loss, i)
            writer.add_scalar('otherloss/sparse_loss', sparsity_loss, i)
            writer.add_scalar('otherloss/cycle_loss', cycle_loss, i)
            writer.add_scalar('otherloss/smooth_loss', smooth_loss, i)
            writer.add_scalar('otherloss/temporal_loss', temporal_loss, i)
            writer.add_scalar('otherloss/uvsmth_loss', uvsmooth_loss, i)
            writer.add_scalar('otherloss/dsmth_loss', dsmooth_loss, i)
            writer.add_scalar('otherloss/gsmth_loss', gsmooth_loss, i)
            writer.add_scalar('otherloss/kpt_loss', kpt_loss, i)
            writer.add_scalar('weight/lr', new_lrate, i)
            writer.add_scalar('weight/uv_loss_weight', new_uv_loss_weight, i)
            writer.add_scalar('weight/alpha_loss_weight', new_alpha_loss_weight, i)
            writer.add_scalar('weight/cycle_loss_weight', new_cycle_loss_weight, i)
            writer.add_scalar('weight/smooth_loss_weight', new_smooth_loss_weight, i)
            writer.add_scalar('weight/temporal_loss_weight', new_temporal_loss_weight, i)
            writer.add_scalar('weight/uvsmooth_loss_weight', new_uvsmooth_loss_weight, i)

        if i % args.i_print == 0:
            print(f"[TRAIN] Iter: {i} Loss: {loss.item()} PSNR: {psnr.item()}")

        if i % args.i_weights == 0:
            path = os.path.join(args.expdir, args.expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                'network_state_dict': nerf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if args.optimize_poses:
                save_dict['rot_raw'] = rot_raw
                save_dict['tran_raw'] = tran_raw
                save_dict['intrin_raw'] = intrin_raw
            torch.save(save_dict, path)
            print('Saved checkpoints at', path)

        if i % args.i_testset == 0:
            print("Save test views")
            savedir = os.path.join(args.expdir, args.expname, 'testset_{:06d}'.format(i))
            os.makedirs(savedir, exist_ok=True)
            dummy_num = ((len(poses) - 1) // args.gpu_num + 1) * args.gpu_num - len(poses)
            poses_tensor = poses.type_as(loss)
            intrinsics_tensor = intrinsics.type_as(loss)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(poses_tensor)
            dummy_intrinsic = intrinsics_tensor[:dummy_num].clone()
            print('render vary t: ', poses.shape, intrinsics.shape)

            with torch.no_grad():
                nerf.eval()
                rgbs, disps = nerf(H, W, t,
                                   poses=torch.cat([poses_tensor, dummy_poses], dim=0),
                                   intrinsics=torch.cat([intrinsics_tensor, dummy_intrinsic], dim=0),
                                   chunk=args.render_chunk)
                rgbs = rgbs[:len(rgbs) - dummy_num]
                disps = disps[:len(disps) - dummy_num]
                rgbs = rgbs.cpu().numpy()
                disps = disps.cpu().numpy()

            for rgb_idx, rgb in enumerate(rgbs):
                imageio.imwrite(os.path.join(savedir, f'rgb_{rgb_idx:03d}.png'), to8b(rgb))
                imageio.imwrite(os.path.join(savedir, f'disp_{rgb_idx:03d}.png'), to8b(disps[rgb_idx]))

        if i % args.i_eval == 0:
            val_ts = [i_ for i_ in range(0, T, T // 6 + 1)]
            print(f"Evaluating on view {test_ids} and time {val_ts}")
            dummy_num = ((len(test_ids) - 1) // args.gpu_num + 1) * args.gpu_num - len(test_ids)
            eval_poses = poses[test_ids].type_as(loss)
            eval_intrinsics = intrinsics[test_ids].type_as(loss)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(eval_poses)
            dummy_intrinsic = eval_intrinsics[:1].clone().expand(dummy_num, 3, 3)
            with torch.no_grad():
                nerf.eval()
                pred_images, gt_images, gt_masks = [], [], []
                for val_t in val_ts:
                    rgbs, _ = nerf(H, W, val_t,
                                   poses=torch.cat([eval_poses, dummy_poses], dim=0),
                                   intrinsics=torch.cat([eval_intrinsics, dummy_intrinsic], dim=0),
                                   chunk=args.render_chunk)
                    rgbs = rgbs[:len(rgbs) - dummy_num]
                    pred_images.append(rgbs)

                    eval_imgpaths = imgpaths[val_t]
                    eval_imgpaths = [eval_imgpaths[i_] for i_ in test_ids]
                    if has_matted_image:
                        rgbas = load_matted(eval_imgpaths)
                        rgbs = torch.tensor(rgbas[..., :3], dtype=torch.float32, device='cpu')
                        msks = torch.tensor(rgbas[..., 3] > 0.5, dtype=torch.float32, device='cpu')
                    else:
                        rgbs = torch.tensor(load_images(eval_imgpaths), dtype=torch.float32, device='cpu')
                        msks = torch.tensor(load_masks(eval_imgpaths), dtype=torch.float32, device='cpu')
                    gt_images.append(rgbs)
                    gt_masks.append(msks)

                pred_images = torch.cat(pred_images, dim=0).clamp(0, 1)
                gt_images = torch.cat(gt_images, dim=0)
                gt_masks = torch.cat(gt_masks)
                test_mse = compute_img_metric(pred_images, gt_images, 'mse', mask=gt_masks)
                test_psnr = compute_img_metric(pred_images, gt_images, 'psnr', mask=gt_masks)
                test_ssim = compute_img_metric(pred_images, gt_images, 'ssim', mask=gt_masks)
                writer.add_scalar("Test/MSE", test_mse, global_step)
                writer.add_scalar("Test/PSNR", test_psnr, global_step)
                writer.add_scalar("Test/SSIM", test_ssim, global_step)

        if i % args.i_video == 0:
            moviebase = os.path.join(args.expdir, args.expname, f'{i:06d}_')
            if hasattr(nerf.module, "texture_map"):
                print('saving texture map')
                if T <= 1:
                    texture_maps = nerf.module.get_texture_map()
                    texture_maps = [tex[0, :3].detach().permute(1, 2, 0).cpu().numpy() for tex in texture_maps]
                    for ti, texture_map in enumerate(texture_maps):
                        imageio.imwrite(moviebase + f"_texture_{ti}.png", to8b(texture_map))
                else:
                    texture_maps = []
                    for ti in range(T):
                        print(f"Get texture map {ti}")
                        texture_map = nerf.module.get_texture_map(t=ti)
                        texture_map = [to8b(tex[0, :3].detach().permute(1, 2, 0).cpu().numpy()) for tex in texture_map]
                        texture_maps.append(texture_map)
                    for ti in range(len(texture_maps[0])):
                        texture_map = [ts[ti] for ts in texture_maps]
                        imageio.mimwrite(moviebase + f"_texture_map_{ti}.mp4", texture_map, fps=30, quality=10)

            print('render poses shape', render_poses.shape, render_intrinsics.shape)
            with torch.no_grad():
                nerf.eval()

                dummy_num = ((len(render_poses) - 1) // args.gpu_num + 1) * args.gpu_num - len(render_poses)
                dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)
                dummy_intrinsic = render_intrinsics[:dummy_num].clone()
                print(f"Append {dummy_num} # of poses to fill all the GPUs")
                # with dynamic camera pose
                if T > 1:
                    rgbs, disps = nerf(H, W,
                                       poses=torch.cat([render_poses, dummy_poses], dim=0),
                                       intrinsics=torch.cat([render_intrinsics, dummy_intrinsic], dim=0),
                                       chunk=args.render_chunk)
                    rgbs = rgbs[:len(rgbs) - dummy_num]
                    disps = disps[:len(disps) - dummy_num]
                    disps = (disps - disps.min()) / (disps.max() - disps.min()).clamp_min(1e-10)
                    rgbs = rgbs.cpu().numpy()
                    disps = disps.cpu().numpy()
                    imageio.mimwrite(moviebase + '_dyn_rgb.mp4', to8b(rgbs), fps=30, quality=10)
                    imageio.mimwrite(moviebase + '_dyn_disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=10)

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()

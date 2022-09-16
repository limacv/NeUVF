import os
import cv2
import json
import glob
import imageio
import numpy as np


def load_masks(imgpaths):
    msklist = []

    for imgdir in imgpaths:
        mskdir = imgdir.replace('images', 'masks').replace('.jpg', '.png')
        msk = imageio.imread(mskdir)

        H, W = msk.shape[0:2]
        msk = msk / 255.0

        # msk   = np.sum(msk, axis=2)
        # msk[msk < 3.0] = 0.0
        # msk[msk == 3.0] = 1.0
        # msk = 1.0 - msk

        newmsk = np.zeros((H, W), dtype=np.float32)
        newmsk[np.logical_and((msk[:, :, 0] == 0), (msk[:, :, 1] == 0), (msk[:, :, 2] == 1.0))] = 1.0

        # imageio.imwrite('newmsk.png', newmsk)
        # print(imgpaths, mskdir, H, W)
        # print(sss)

        msklist.append(newmsk)

    msklist = np.stack(msklist, 0)

    return msklist


def has_matted(imgpaths):
    exampledir = imgpaths[-1].replace('images', 'images_rgba').replace('.jpg', '.png')
    return os.path.exists(exampledir)


def load_matted(imgpaths):
    imglist = []
    for imgdir in imgpaths:
        imgdir = imgdir.replace('images', 'images_rgba').replace('.jpg', '.png')
        rgba = imageio.imread(imgdir)
        assert rgba.shape[-1] == 4, "cannot load rgba png"
        rgba = rgba / 255.0
        rgba[..., :3] = rgba[..., :3] * rgba[..., 3:4]
        imglist.append(rgba)

    imglist = np.stack(imglist, 0)
    return imglist


def load_images(imgpaths):
    imglist = []

    for imgdir in imgpaths:
        img = imageio.imread(imgdir)
        img = img / 255.0
        imglist.append(img)

    imglist = np.stack(imglist, 0)

    return imglist


def load_imgpaths(datadir, factor, frm_num, frm_start=0):
    imgdir = datadir + f'/images_{factor}x'
    cams = sorted(os.listdir(imgdir))

    if frm_num == -1:
        imgs = sorted(os.listdir(imgdir + '/' + cams[0]))
    else:
        imgs = sorted(os.listdir(imgdir + '/' + cams[0]))[frm_start:frm_start + frm_num]

    imgpaths = []

    for img in imgs:
        camlist = []

        for cam in cams:
            camlist.append(os.path.join(imgdir, cam, img))

        imgpaths.append(camlist)

    return imgpaths


def load_params(params_path, factor):
    with open(params_path, 'r') as fp:
        camera_json = json.load(fp)

    extlist = []
    dislist = []
    intlist = []

    for cam in sorted(camera_json):

        orientation = np.asarray(camera_json[cam]['orientation'])
        position = np.asarray(camera_json[cam]['position'])
        # distortion = np.asarray(camera_json[cam]['distortion'])
        distortion = np.asarray(camera_json[cam]['intrinsic'])
        intrinsic = np.asarray(camera_json[cam]['intrinsic'])

        if factor != 1:
            intrinsic[0:2, 0:3] = intrinsic[0:2, 0:3] * 1.0 / factor

        extlist.append(np.concatenate((orientation, position[:, np.newaxis]), axis=1))
        dislist.append(distortion)
        intlist.append(intrinsic)

    extlist = np.stack(extlist, 0)
    dislist = np.stack(dislist, 0)
    intlist = np.stack(intlist, 0)

    return extlist, dislist, intlist


def load_bounds(params_path):
    with open(params_path, 'r') as fp:
        camera_json = json.load(fp)

    coord_min = np.array(camera_json['min']).astype(np.float32)
    coord_max = np.array(camera_json['max']).astype(np.float32)
    center = (coord_min + coord_max) / 2
    radius = np.linalg.norm(center - coord_min)
    # boundlist = []
    #
    # for cam in sorted(camera_json):
    #     near = np.asarray(float(camera_json[cam]['near']))
    #     far = np.asarray(float(camera_json[cam]['far']))
    #     boundlist.append([near, far])
    #
    # boundlist = np.stack(boundlist, 0)
    return center, radius


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(vec1, vec2, center):
    vec0 = normalize(np.cross(vec1, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, center], 1)

    return m


def poses_avg(poses):
    vec1 = normalize(poses[:, 0:3, 1].sum(0))
    vec2 = normalize(poses[:, 0:3, 2].sum(0))
    center = poses[:, 0:3, 3].mean(0)
    c2w = viewmatrix(vec1, vec2, center)

    return c2w


def recenter_poses(poses):
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)

    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses

    return poses[:, :3, :4], np.linalg.inv(c2w)


def recenter_poses_with_center(poses, center):
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w[:3, 3] = center

    c2w = np.concatenate([c2w[:3, :4], bottom], -2)

    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses

    return poses[:, :3, :4], np.linalg.inv(c2w)


def load_data(datadir, factor, frm_num, frm_start=0, bd_factor=0.75, recenter=True):
    imgpaths = load_imgpaths(
        datadir, factor, frm_num=frm_num, frm_start=frm_start)  # M * N * H * W * C

    print("Load params from poses_bounds.npz")
    data = np.load(os.path.join(datadir, 'poses_bounds.npz'))
    poses = data['poses']
    intrinsics = data['intrinsics']

    intrinsics[:, 0:2, 0:3] = intrinsics[:, 0:2, 0:3] / factor

    box_min, box_max = data['box_min'], data['box_max']
    center = (box_min + box_max) / 2
    box_min = center + (box_min - center) / bd_factor
    box_max = center + (box_max - center) / bd_factor
    bds = np.stack([box_min, box_max])

    avg_pose = poses_avg(poses)
    avg_pose[:3, 3] *= 1.1

    up = normalize(poses[:, :3, 1].sum(0))
    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    rads[0] *= 0.9
    rads[1] *= 0
    focal = 0

    # N = len(imgpaths)
    N = 120
    render_poses = render_path_spiral(avg_pose, up, rads, focal, zrate=1, zdelta=0.5, rots=0.5, N=N)
    render_intrinsics = np.repeat(intrinsics[:1], N, axis=0)

    return imgpaths, poses.astype(np.float32), intrinsics.astype(np.float32), \
           bds, render_poses.astype(np.float32), render_intrinsics.astype(np.float32)


def load_position_maps(datadir, factor, basenames):
    root = os.path.join(datadir, f"prnet_*x", "global")
    root = glob.glob(root)
    assert len(root) > 0, f"cannot find position maps in the datadir {datadir}, " \
                                f"the position maps are supposed to save under " \
                                f"<datadir>/prnet_<factor>x/global"
    root = root[0]
    print(f"using position map results from {root}")

    pos_maps = []
    for i in range(len(basenames)):
        uv_gt = np.load(os.path.join(root, f"{basenames[i]}.npy"))
        uv_gt = np.concatenate([uv_gt[:, :3], uv_gt[:, 4:5], uv_gt[:, 3:4]], axis=-1)
        pos_maps.append(uv_gt)

    return pos_maps


def render_path_spiral(c2w, up, rads, focal, zrate, zdelta, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        # view direction
        # c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), (np.cos(theta * zrate) * zdelta) ** 2, 1.]) * rads)
        # camera poses
        z = normalize(c - np.array([0, 0, -focal]))
        render_poses.append(viewmatrix(up, -z, c))
    return np.stack(render_poses)

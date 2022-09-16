import configargparse


def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--datadir", type=str,
                        help='input data directory')
    parser.add_argument("--expdir", type=str,
                        help='where to store ckpts and logs')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--seed", type=int, default=666,
                        help='random seed')
    parser.add_argument("--gpu_num", type=int, default='-1', help='number of processes')
    parser.add_argument("--frm_num", type=int, default=-1, help='number of frames to use')
    parser.add_argument("--frm_start", type=int, default=0, help='starting frames')
    parser.add_argument("--bd_factor", type=float, default=0.65, help='expand factor of the ROI box')
    parser.add_argument("--nerf_type", type=str, default='NeUVFModulateT',
                        choices=['NeUVFModulateT', 'NeRFModulateT', 'NeRFTemporal'],
                        help='use the original raw2output (not forcing the last layer to be alpha=1')

    ## data options
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--test_ids", type=str, default='4',
                        help='example: 3,4,5,6  splited by \',\'')

    # training options
    parser.add_argument("--N_iters", type=int, default=50000)
    parser.add_argument("--itertions_per_frm", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--masked_sample_precent", type=float, default=0.92,
                        help="in batch_size samples, precent of the samples that are sampled at"
                             "masked region, set to 1 to disable the samples on black region")
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--optimize_poses", default=False, action='store_true',
                        help='optimize camera poses jointly')
    parser.add_argument("--optimize_poses_start", type=int, default=0, help='start step of optimizing poses')
    parser.add_argument("--promote_fuse_texture_step", type=int, default=1e15,
                        help='we first use MLP to model explicit texture for better training. '
                             'Later we generate a texture map based on the MLP')
    parser.add_argument("--freeze_uvfield_step", type=int, default=1e15,
                        help='iteration that starts to freeze the uv field')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64,
                        help='number of additional fine samples per ray')
    parser.add_argument("--N_samples_fine", type=int, default=64,
                        help='n sample fine = N_samples_fine + N_importance')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter the sampling')
    parser.add_argument("--raw_noise_std", type=float, default=1e0,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_canonical", action='store_true',
                        help='if true, the DNeRF is like traditional NeRF. Not used for NeUVF')
    parser.add_argument("--render_chunk", type=int, default=1024,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--use_two_models_for_fine", action='store_true', default=True,
                        help='if true, nerf_coarse == nerf_fine')
    parser.add_argument("--not_supervise_rgb0", action='store_true', default=False,
                        help='if true, rgb0 well not considered as part of the loss')
    parser.add_argument("--best_frame_idx", type=int, default=-1,
                        help='if > 0, the first epoch will be trained only on this frame. Used for initialization')

    # Rendering configuration for render.py
    parser.add_argument("--render_rgba", action='store_true',
                        help='render images with alpha channel')
    parser.add_argument("--render_keypoints", action='store_true',
                        help='render keypoints')

    # logging options
    parser.add_argument("--i_tensorboard",    type=int, default=300,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_print",   type=int, default=300,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=20000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=20000,
                        help='frequency of testset saving')
    parser.add_argument("--i_eval", type=int, default=10000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=10000,
                        help='frequency of render_poses video saving')

    # the base network
    # ====================================
    parser.add_argument("--netdepth", type=int, default=6,
                        help='layers of the base network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--multires", type=int, default=9,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--use_viewdirs", type=bool, default=True,
                        help='the pipeline depend on view dir or not')
    parser.add_argument("--embed_type", type=str, default='pe',
                        help='embed type of spatial coordinates. choose among pe, none, hash, dict')
    parser.add_argument("--log2_embed_hash_size", type=int, default=19,
                        help='deprecated configure')
    parser.add_argument("--multires_window_start", type=int, default=0,
                        help='(for spatial coordiantes) windowed PE start step')
    parser.add_argument("--multires_window_end", type=int, default=-1,
                        help='(for spatial coordiantes) windowed PE end step, negative to disable')
    parser.add_argument("--time_embed_type", type=str, default="latent",
                        help='choose among pe or latent')
    parser.add_argument("--time_multires", type=int, default=5,
                        help='embedding dim on time axis if time_embed_type == pe')
    parser.add_argument("--latent_size", type=int, default=0,
                        help='latent_size if time_embed_type == latent')
    parser.add_argument("--rgb_activate", type=str, default='sigmoid',
                        help='activate function for rgb output, choose among "none", "sigmoid"')
    parser.add_argument("--sigma_activate", type=str, default='relu',
                        help='activate function for sigma output, choose among "relu", "softplus",'
                             '"volsdf"')

    ## For UVField MLP
    parser.add_argument("--uv_activate", type=str, default='tanh',
                        help='activate function for uv output, choose among "tanh"')
    parser.add_argument("--uv_map_gt_skip_num", type=int, default=0,
                        help='will only load pos map every <skip_num> frames, usually set to 0')
    parser.add_argument("--uv_batch_size", type=int, default=10240,
                        help='batch size of the uv supervision')
    parser.add_argument("--uv_loss_weight", type=float, default=0,
                        help='uv loss weight')
    parser.add_argument("--uv_loss_decay", type=int, default=10,
                        help='uv loss decay, usually a small value since we use for initialization only')
    parser.add_argument("--uv_loss_noise_std", type=float, default=0.002,
                        help='add randn noise to the supervision location of the uv_loss')
    parser.add_argument("--uv_map_face_roi", type=float, default=0.64,
                        help='percent of the face texture in the all texture')

    # The texture MLP
    # ================================
    parser.add_argument("--multires_views", type=int, default=3,
                        help='view positional encoding for the view-dependent texture MLP')
    parser.add_argument("--multires_views_window_start", type=int, default=0,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_views_window_end", type=int, default=-1,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--use_two_texmodels_for_fine", type=bool, default=True,
                        help='if true, nerf_coarse == nerf_fine')
    parser.add_argument("--texture_type", type=str, default="map",
                        help='type of texure, choose among map, mlp, fuse')
    # For texture_type == map
    parser.add_argument("--texture_map_channel", type=int, default=3,
                        help='number of channels in the texture map, we use 3 to represent RGB')
    parser.add_argument("--texture_map_resolution", type=int, default=1024,
                        help='number of channels in the texture map, negative to disable')
    parser.add_argument("--texture_map_gradient_multiply", type=int, default=10,
                        help='gradient of the texture map will be multiply by this value, '
                             'sometimes this yields better texture')
    parser.add_argument("--texture_map_ini", type=str, default='',
                        help='initialization of the texture map')
    # For texture_type == mlp
    parser.add_argument("--texnetdepth", type=int, default=6,
                        help='layers in network')
    parser.add_argument("--texnetwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--texnet_view_layeridx", type=int, default=-2,
                        help='the layer index when introducing view embedding')
    parser.add_argument("--texnet_time_layeridx", type=int, default=0,
                        help='the layer index when introducing time embedding')
    parser.add_argument("--tex_multires", type=int, default=12,
                        help='embedding for texture mlp')
    parser.add_argument("--tex_embed_type", type=str, default='pe',
                        help='type of tex embed')
    parser.add_argument("--tex_log2_hash_size", type=int, default=14,
                        help='deprecated now')
    parser.add_argument("--use_two_time_for_tex", type=bool, default=True,
                        help='use two time embedding for the texture mlp')
    parser.add_argument("--time_embed_type_for_tex", type=str, default="latent",
                        help='pe or latent')
    parser.add_argument("--latent_size_for_tex", type=int, default=192,
                        help='latent_size for texture mlp')
    parser.add_argument("--time_multires_for_tex", type=int, default=5,
                        help='embedding dim on time axis, not use if time_embed_type_for_tex==pe')
    parser.add_argument("--time_multires_window_start_for_tex", type=int, default=0,
                        help='not used in the paper')
    parser.add_argument("--time_multires_window_end_for_tex", type=int, default=-1,
                        help='not used in the paper')

    # Towards geometry editing (the explicit warping module)
    # ================================
    parser.add_argument("--density_type", type=str, default='direct',
                        help='choose among direct, xyz_norm, not used in the paper')
    parser.add_argument("--explicit_warp_type", type=str, default='kptaffine',
                        help='choose among none, proj, kptaffine')
    parser.add_argument("--canonicaldir", type=str, default='assets/canonical_vertices_my.npy',
                        help='path to canonical dir, which contains vertices position in canonical space')
    parser.add_argument("--kptidsdir", type=str, default='assets/kpts2.npy',
                        help='index of landmarks that is manually picked from all the vertices')
    parser.add_argument("--uvweightdir", type=str, default='assets/face_uv_mask.png',
                        help='path to canonical dir')
    parser.add_argument("--model_affine", type=bool, default=False,
                        help='whether to fit per control point affine transform')
    parser.add_argument("--rbf_perframe", type=bool, default=True,
                        help='whether to fit per frame rbf')

    ## Losses
    # ========================
    parser.add_argument("--sparsity_type", type=str, default='none',
                        help='sparsity loss type, choose among none, l1, l1/l2, entropy')
    parser.add_argument("--sparsity_loss_weight", type=float, default=0,
                        help='sparsity loss weight')
    parser.add_argument("--sparsity_loss_start_step", type=float, default=50000,
                        help='sparsity_loss_start_step')

    parser.add_argument("--cycle_loss_weight", type=float, default=0,
                        help='cycle loss of neutex')
    parser.add_argument("--cycle_loss_decay", type=float, default=0,
                        help='cycle loss of neutex')
    parser.add_argument("--cyclenetdepth", type=int, default=5,
                        help='layers in network')
    parser.add_argument("--cyclenetwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--cyclenet_time_layeridx", type=int, default=0,
                        help='the layer index when introducing time embedding')
    parser.add_argument("--use_two_time_for_cycle", type=bool, default=True,
                        help='pe or latent')
    parser.add_argument("--time_embed_type_for_cycle", type=str, default="latent",
                        help='pe or latent')
    parser.add_argument("--latent_size_for_cycle", type=int, default=0,
                        help='latent_size')
    parser.add_argument("--time_multires_for_cycle", type=int, default=5,
                        help='embedding dim on time axis')
    parser.add_argument("--use_two_embed_for_cycle", type=bool, default=True,
                        help='use two individual embedding for input xyz')
    parser.add_argument("--embed_type_for_cycle", type=str, default='pe',
                        help='embed type for cycle')
    parser.add_argument("--multires_for_cycle", type=int, default=2,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--log2_hash_size_for_cycle", type=int, default=14,
                        help='deprecated now')

    parser.add_argument("--alpha_type", type=str, default="multiply",
                        help='choose among add, multiply')
    parser.add_argument("--alpha_loss_weight", type=float, default=0,
                        help='alpha sparsity loss')
    parser.add_argument("--alpha_loss_decay", type=int, default=1e10,
                        help='alpha sparsity loss decay')

    parser.add_argument("--smooth_loss_weight", type=float, default=0,
                        help='edge-aware disparity smooth loss, not use in the paper')
    parser.add_argument("--smooth_loss_start_decay", type=int, default=1e10,
                            help='edge-aware disparity smooth loss start step')

    parser.add_argument("--temporal_loss_weight", type=float, default=0,
                        help='temporal consistency loss, not use in the paper')
    parser.add_argument("--temporal_loss_start_step", type=float, default=0,
                        help='temporal consistency loss start step')
    parser.add_argument("--temporal_loss_patch_num", type=int, default=100,
                        help='patch number of temporal loss, the actual number will multiply by the gpu_num')

    parser.add_argument("--uvsmooth_loss_weight", type=float, default=0,
                        help='smoothness for the uv field')
    parser.add_argument("--uvprepsmooth_loss_weight", type=float, default=0,
                        help='smoothness for the uv field project to surface')
    parser.add_argument("--dsmooth_loss_weight", type=float, default=0,
                        help='smoothness for the density raw field')

    parser.add_argument("--kpt_loss_weight", type=float, default=0,
                        help='keypoint loss weight')
    parser.add_argument("--gsmooth_loss_weight", type=float, default=0,
                        help='explicit geometry parameter loss weight')
    parser.add_argument("--gsmooth_loss_type", type=str, default='o2',
                        help='o1 or o2, order n derivative')
    parser.add_argument("--gsmooth_loss_decay", type=int, default=1e10,
                        help='make the explicit warp more temporal smooth')

    # For DNeRF
    parser.add_argument("--use_two_dmodels_for_fine", type=bool, default=True,
                        help='if true, nerf_coarse == nerf_fine')
    parser.add_argument("--dnetdepth", type=int, default=7,
                        help='layers in network')
    parser.add_argument("--dnetwidth", type=int, default=128,
                        help='channels per layer')
    parser.add_argument("--ambient_slicing_dim", type=int, default=0,
                        help='ambient slicing dim, set to non-zero to mimic hyperNeRF')
    parser.add_argument("--slice_multires", type=int, default=7,
                        help='embedding for ambient slicing')
    parser.add_argument("--slicenetdepth", type=int, default=6,
                        help='used if ambient_slicing_dim > 0')
    parser.add_argument("--slicenetwidth", type=int, default=128,
                        help='used if ambient_slicing_dim > 0')
    return parser

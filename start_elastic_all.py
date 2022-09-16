import os
from glob import glob

files = [
    # 4-28 night submit => nouv
    # "ours_full_nouv/chuangxinb_ours_full",
    # "ours_full_nouv/chuangxinc_ours_full",
    # "ours_full_nouv/cjujuc_ours_full",
    # "ours_full_nouv/cmalib_ours_full",
    # "ours_full_nouv/cqihanga_ours_full",
    # "ours_full_nouv/cweiyub_ours_full",
    # "ours_full_nouv/cxiaoyu2a_ours_full",

    # 5-1 noon submit => nouv
    # "ours_full_biguv/chuangxinb_ours_full",
    # "ours_full_biguv/cjujuc_ours_full",
    # "ours_full_biguv/cmalib_ours_full",
    # "ours_full_biguv/cweiyub_ours_full",
    #
    # "ours_full_noalpha/chuangxinb_ours_full",
    # "ours_full_noalpha/cjujuc_ours_full",
    # "ours_full_noalpha/cmalib_ours_full",
    # "ours_full_noalpha/cweiyub_ours_full",
    #
    # "ours_full_1stage/chuangxinb_ours_full",
    # "ours_full_1stage/cjujuc_ours_full",
    # "ours_full_1stage/cmalib_ours_full",
    # "ours_full_1stage/cweiyub_ours_full",
    #
    # "ours_full_nogsmth/chuangxinb_ours_full",
    # "ours_full_nogsmth/cjujuc_ours_full",
    # "ours_full_nogsmth/cmalib_ours_full",
    # "ours_full_nogsmth/cweiyub_ours_full",
    #
    # "ours_full_nouvreg/chuangxinb_ours_full",
    # "ours_full_nouvreg/cjujuc_ours_full",
    # "ours_full_nouvreg/cmalib_ours_full",
    # "ours_full_nouvreg/cweiyub_ours_full",

    # 5-2 night submit
    # "ours_full_geometry/chuangxinc_nogsmth",
    # "ours_full_geometry/chuangxinc_noseman",
    # "ours_full_geometry/chuangxinc_nowarp",
    # "ours_full_geometry/cweiyub_nogsmth",
    # "ours_full_geometry/cweiyub_noseman",
    # "ours_full_geometry/cweiyub_nowarp",
    # "ours_full_geometry/cxiaoyu2a_nogsmth",
    # "ours_full_geometry/cxiaoyu2a_noseman",
    # "ours_full_geometry/cxiaoyu2a_nowarp",

    # 5-4/5-8 night submit => 91
    # "ours_full1/chenqin2a_ours_full",
    # "ours_full1/chuangxinb_ours_full",
    # "ours_full1/cjujuc_ours_full",
    # "ours_full1/cmali2a_ours_full",
    # "ours_full1/cmalib_ours_full",
    # "ours_full1/cmalic_ours_full",
    # "ours_full1/cweiyub_ours_full",
    # "ours_full1/cxianjin_ours_full",
    # "ours_full1/cxiaoyu2a_ours_full",
    # "ours_full1/dmali2a_ours_full",
    # "ours_full1/cqihanga_ours_full",
    # "ours_full1/chuangxinc_ours_full",

    # 5-10 night submit
    # "ours_full_nouvreg/cweiyub_ours_full_uvreg0001",
    # "ours_full_nouvreg/cweiyub_ours_full_uvreg0005",
    # "ours_full_nouvreg/cweiyub_ours_full_uvreg001",
    # "ours_full_nouvreg/cweiyub_ours_full_uvreg01",
    # "ours_full_nouvreg/cweiyub_ours_full_uvreg05",
    # "ours_full_nouvreg/cweiyub_ours_full_uvreg5",
    # "ours_full_nouvreg/cweiyub_ours_full_uvreg1",
    # "ours_full_nouvreg/cweiyub_ours_full_uvreg10",

    # "ours_full_noalpha/cweiyub_ours_full_alpha0001",
    # "ours_full_noalpha/cweiyub_ours_full_alpha0005",
    # "ours_full_noalpha/cweiyub_ours_full_alpha001",
    # "ours_full_noalpha/cweiyub_ours_full_alpha01",
    # "ours_full_noalpha/cweiyub_ours_full_alpha05",
    # "ours_full_noalpha/cweiyub_ours_full_alpha5",
    # "ours_full_noalpha/cweiyub_ours_full_alpha1",
    # "ours_full_noalpha/cweiyub_ours_full_alpha10",

    # 5-11 morning submit
    # "ours_alphablend/chuangxinb_ours_full",
    # "ours_alphablend/cjujuc_ours_full",

    # "ours_full_smallembed/chuangxinb_ours_full",
    # "ours_full_smallembed/cjujuc_ours_full",
    # "ours_full_smallembed/cmalib_ours_full",
    # "ours_full_smallembed/cweiyub_ours_full",
    # "ours_full_noalpha/cmalib_ours_full_alpha05",
    # "ours_full_noalpha/cmalib_ours_full_alpha0005",
    # "ours_nofgloss/chenqin2a_ours_full",

    # "ours_full_uvploss/chuangxinb_ours_full",
    # "ours_full_uvploss/cjujuc_ours_full",
    # "ours_full_uvploss/cmalib_ours_full",
    # "ours_full_uvploss/cweiyub_ours_full",

    # "ours_full_nouvreg/chuangxinb_ours_full_uvreg01",
    # "ours_full_nouvreg/chuangxinb_ours_full_uvreg001",
    # "ours_full_nouvreg/chuangxinb_ours_full_uvreg0001",
    # "ours_full_nouvreg/chuangxinb_ours_full_uvreg05",
    # "ours_full_nouvreg/chuangxinb_ours_full_uvreg0005",

    # "exp_n_views/chuangxinb_3views",
    # "exp_n_views/chuangxinb_5views",
    # "exp_n_views/chuangxinb_7views",
    # "exp_n_views/chuangxinb_9views",

    # "facescape/subject1",
    # "facescape/subject3",
    # "facescape/subject4",
    # "facescape/subject5",
    # "facescape/subject6",
    # "facescape/subject7",
    # "facescape/subject8",
    # "facescape/subject9",
    # "facescape/subject10",
    # "facescape/subject12",
    # "facescape/subject13",
    # "facescape/subject16",
    # "facescape/subject19",

    # "facescape/subject21",
    # "facescape/subject22",
    # "facescape/subject23",
    # "facescape/subject24",
    # "facescape/subject26",
    # "facescape/subject27",
    # "facescape/subject29",
    # "facescape/subject30",
    # "facescape/subject31",
    # "facescape/subject33",
    # "facescape/subject34",

    # "facescape/subject35",
    # "facescape/subject36",
    # "facescape/subject37",
    # "facescape/subject38",
    # "facescape/subject43",
    # "facescape/subject44",
    # "facescape/subject45",
    # "facescape/subject46",
    # "facescape/subject47",
    # "facescape/subject48",
    # "facescape/subject51",
    # "facescape/subject52",
    # "facescape/subject53",
    # "facescape/subject55",
    # "facescape/subject56",
    # "facescape/subject57",

    # "facescape/subject58",
    # "facescape/subject59",
    # "facescape/subject60",
    # "facescape/subject61",
    # "facescape/subject63",
    # "facescape/subject64",
    # "facescape/subject65",
    # "facescape/subject66",
    # "facescape/subject67",
    # "facescape/subject69",
    # "facescape/subject70",
    # "facescape/subject71",
    # "facescape/subject72",
    # "facescape/subject73",
    # "facescape/subject74",
    # "facescape/subject76",
    # "facescape/subject77",
    # "facescape/subject79",
    # "facescape/subject80",
    # "facescape/subject81",
    # "facescape/subject82",
    # "facescape/subject85",
    # "facescape/subject86",
    # "facescape/subject87",
    # "facescape/subject89",
    # "facescape/subject90",
    # "facescape/subject93",
    # "facescape/subject94",
    # "facescape/subject95",
    # "facescape/subject96",
    # "facescape/subject98",
    # "facescape/subject99",

]

files = [file + ".txt" for file in files]

startjson = """
{
  "Token": "loCr7k16k3DEY7283MuFqA",
  "business_flag": "TEG_AILab_CVC_DigitalContent",
  "model_local_file_path": "/apdcephfs/private_leema/makeuptrans",
  "enable_evicted_pulled_up": true,
  "is_elasticity": true,
  "host_num": 1,
  "host_gpu_num": <gpu_num>,
  "GPUName": "A100,V100",
  "image_full_name": "mirrors.tencent.com/leema/nerf-pytorch:2",
  "init_cmd": "jizhi_client mount -l cq ~/private_leema",
  "start_cmd": "python3 -u train.py --config configs/<file_base> --i_weights 10000 && python3 -u train.py --config configs/<file_base> --gpu_num 1 --render_only --texture_map_post /apdcephfs/private_leema/data/NeRFtx/textures/checker.png ",
  "task_flag": "<task_name>"
}
"""
for file in files:
    with open(os.path.join("configs", file), 'r') as f:
        lines = f.read().splitlines()
    line = [l_ for l_ in lines if "expname" in l_][0]
    line = line[line.find("=") + 1:]
    line = line.lstrip(' ').strip('\n') + "_elastic"

    gpuline = [l_ for l_ in lines if "gpu_num" in l_][0]
    gpuline = gpuline[gpuline.find("=") + 1:]
    gpuline = gpuline.lstrip(' ').strip('\n')
    gpuline = int(gpuline)
    print(f"Launching expname = {line} with gpunum = {gpuline}")
    with open("_tmp_start.json", 'w') as f:
        json = startjson.replace("<gpu_num>", f"{gpuline}")\
            .replace("<file_base>", file)\
            .replace("<task_name>", line)
        f.write(json)
    os.system("jizhi_client start -scfg _tmp_start.json")

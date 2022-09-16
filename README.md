This is the official implementation of the paper [Neural Parameterization for Dynamic Human Head Editing](https://arxiv.org/abs/2207.00210)
# Neural Parameterization

We present a method for intuitively and consistently editing the geometry and appearance of a 3D human head. 

![](https://limacv.github.io/NeUVF/teasers/teaser.png)

## Method Overview

![](https://limacv.github.io/NeUVF/teasers/pipeline.png)

We use NeRF-like representations for modeling the human head. \
To edit the appearance, we extend the idea of [NeuTex](https://arxiv.org/abs/2103.00762), by texture mapping the color
of a dynamic NeRF to a 2D texture. For geometry editing, we use several semantically-rich facial landmarks to generate a 3D 
deformation field, so that by editing the keypoints, users are able to consistently edit the shape of the head. 

We also provide a simple user interface for interactively editing the appearance and geometry. The source code of the UI
can be find in [here](???)

## Quick Start

### 1. Install environment

```
git clone https://github.com/limacv/NeUVF.git
cd NeUVF
pip install -r requirements.txt
```
<details>
  <summary> Dependencies (click to expand) </summary>

   - numpy
   - scikit-image
   - torch>=1.8
   - torchvision>=0.9.1
   - imageio
   - imageio-ffmpeg
   - matplotlib
   - configargparse
   - tensorboardX>=2.0
   - opencv-python
   - trimesh
</details>

### 2. Download sample data
Due to privacy issue, we could only provide two sample data. The link can be find [here](???). 

### 3. Setting parameters
Changing the data path and log path in the ```configs/demo_blurball.txt```

### 4. Execute

```
python3 run_nerf.py --config configs/demo_blurball.txt
```
This will generate MLP weights and intermediate results in ```<basedir>/<expname>```, and generate tensorboard files in ```<tbdir>```

## Some Notes

### GPU memory
Due to the NeRF-like representation. In our representation, we use 3 V100 GPUs, and the training takes about 24h. 
If you have fewer GPUs, the training time may be longer. If you customize your batch_size, please make sure that 
it is divisible by 2*<gpu_num>

### D-NeRF, HyperNeRF, DynNeRF baseline
The code also contains implementation of [D-NeRF](https://arxiv.org/abs/2011.13961), 
[HyperNeRF](https://hypernerf.github.io/) and [DynNeRF](https://neural-3d-video.github.io/). 
These are not official implementations, so details maybe different. For example, 
for HyperNeRF, we do not implement the Windowed-Positional-Encoding, and the ambient slicing surface is shared 
for the entire volume instead of each sampling location.

## Limitation
Due to the parameterization of the color, our method sometimes could not achieve the same level of reconstruction 
accuracy as other dynamic NeRF methods, which focus only on reconstruction, as illustrated in the experiment section.
Besides, the geometry editing module is somewhat 

## Citation
If you find this useful, please consider citing our paper:
```
???
```

## Acknowledge
We would like to thank individuals who generously provide the face data for this project. 
The skeleton of this code is derived from the famous pytorch reimplementation of NeRF, [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch/). 
We appreciate the effort of the contributor to that repository.

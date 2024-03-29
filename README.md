This is the official implementation of the paper [Neural Parameterization for Dynamic Human Head Editing](https://arxiv.org/abs/2207.00210).
In this work we present a method for intuitively and consistently editing the geometry and appearance of a 3D human head. The website can be found [here](https://limacv.github.io/neuvf_web/).

![](https://limacv.github.io/NeUVF/teasers/teaser.png)

## Method Overview

![](https://limacv.github.io/NeUVF/teasers/pipeline.png)

We use NeRF-like representations for modeling the human head. 
To edit the appearance, we extend the idea of [NeuTex](https://arxiv.org/abs/2103.00762), by texture mapping the color
of a dynamic NeRF to a 2D texture. For geometry editing, we use several semantically-rich facial landmarks to generate a 3D 
deformation field, so that by editing the keypoints, users are able to consistently edit the shape of the head. 

We also provide a simple user interface for interactively editing the appearance and geometry. 

## Run the code

### 1. Install environment

```
git clone https://github.com/limacv/NeUVF.git
cd NeUVF
pip install -r requirements.txt
```

### 2. Download sample data
Due to privacy issue, we could only provide two sample data. The data can be find [here](https://drive.google.com/drive/folders/1W_xEq4mJgJOFsTl1Ra9DFMXrfTeeJwmJ?usp=sharing). 

### 3. Prepare config files
We provide several config files in the ```config``` folder. 
Change the ```datadir``` and ```expdir``` in the config file to your data and log folder.

### 4. Train a NeRF representation
```
python train.py --config <config_files>
```
This will generate MLP weights and 
intermediate results in ```<basedir>/<expname>```. 

### 5. Rendering a NeRF after training
 ```
python render.py --config <config_files> --<other_config>
```
This will render specific view and time sequence in the ```<basedir>/<expname>/render_only_images```.
There are a bunch of configs (```--render_*```) that control the rendering process. 

### 6. Exporting coarse mesh and texture for editing
```
python script_export_obj.py --config <config_files> \
  --render_canonical
```
This will generate mesh and texture files in the ```<basedir>/<expname>/mesh```.
We specify ```render_canonical``` since we want to extract canonical mesh

### 7. Running UI and simple edit
The simple UI is based on PyOpenGL, glfw and PyImgui. 
It should be simple to install the required dependency. 
After install all the dependencies, run
```
cd UI
python main.py
```
In Windows, if you have python installed under PATH, just double click the main.py.
This will open a window for editing and saving. 

### 8. Rendering the edited NeRF
After editing the texture or the geometry, one can render the edited representation
by again using the ```render.py```:
```
python render.py --config <config_files> \
  --texture_map_post <path_to_new_texture> \
  --texture_map_post_isfull \
  --render_deformed <path_to_npz_files_from_UI>
```

## Some Notes

### GPU memory
Due to the NeRF-like representation. In our representation, we use 3 V100 GPUs, and the training takes about 24h. 
If you have fewer GPUs, the training time may be longer. If you customize your batch_size, please make sure that 
it is divisible by 2*<gpu_num>.

### D-NeRF, HyperNeRF, DynNeRF baseline
The code also contains implementation of [D-NeRF](https://arxiv.org/abs/2011.13961), 
[HyperNeRF](https://hypernerf.github.io/) and [DynNeRF](https://neural-3d-video.github.io/). 
These are not official implementations, so details maybe different. For example, 
for HyperNeRF, we do not implement the Windowed-Positional-Encoding, and the ambient slicing surface is shared 
for the entire volume instead of each sampling location.

## Limitation
Due to the parameterization of the color, our method sometimes could not achieve the same level of reconstruction 
accuracy as other dynamic NeRF methods, which focus only on reconstruction.
Besides, the geometry editing module is somewhat simple and cannot achieve complex shape editing 
such as topology changes. More details please read the paper. 

## Citation
If you find this useful, please consider citing our paper:
```
TODO
```

## Acknowledge
We would like to thank individuals who generously provide the face data for this project. 
The skeleton of this code is derived from the famous pytorch reimplementation of NeRF, [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch/). 
We appreciate the effort of the contributor to that repository.

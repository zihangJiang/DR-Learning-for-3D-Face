# Disentangled Representation Learning for 3D Face Shape

This repository is the implementation of our CVPR 2019 paper <a href="https://arxiv.org/abs/1902.09887">"Disentangled Representation Learning for 3D Face Shape"</a>

Authors: [Zihang Jiang](https://zihangjiang.github.io/) ,[Qianyi Wu](https://qianyiwu.github.io), [Keyu Chen](https://kychern.github.io/) and [Juyong Zhang](http://staff.ustc.edu.cn/~juyong/) .
<p align="center">
<img src = "Pict/manifold.png" height = "400px"/>
</p>

# Our Proposed Framework

<img src = "Pict/pipeline.png" height = "400px"/>

# Result Examples
<img src = "Pict/interpolation.png" height = "400px"/> 
We can manipulate 3D face shape in expression and identity code space.


# Usage

### Dataset
Please download [FaceWareHouse](http://kunzhou.net/zjugaps/facewarehouse/) dataset.

### Requirements
#### 1. Basic Environment
tensorflow-gpu = 1.9.0
Keras = 2.2.2
openmesh
#### 2. Requirements for Data  Processing (About Deformation Representation Feature)
1.  We provide a python interface for obtaining ***Deformation Representation (DR) Feature***. Code are avaliable at [Here](https://github.com/QianyiWu/get_dr_py) to generate DR feature for each obj file by specific one reference mesh. After that, you can change the data_path and data_format in `src/data_utils.py`.

2. To recover mesh from DR feature, you need to compile [get_mesh](https://github.com/QianyiWu/get_mesh_py_API), and replace the `get_mesh.cpython-36m-x86_64-linux-gnu.so` in `src` folder.

3. Also, python version of [libigl](https://github.com/libigl/libigl) is needed for mesh-IO and you need to replace the `pyigl.so` in `src` folder

After all requirements are satisfied, you can use following command to train and test the model.
### Training 

Run following command to generate training and testing data for 3D face DR learning
```bash
python src/data_utils.py
```
We have provided **DR feature of expression mesh on Meanface** we used in this project on [here](https://drive.google.com/open?id=1GgCKnKRrLR8r51Pw_TBqDHK8vdu6Oj4M).


Run this command to pretrain identity branch
```bash
python main.py -m gcn_vae_id -e 20
```

Run following command to pretrain expression branch
```bash
python main.py -m gcn_vae_exp -e 20
```

Run following command for end_to_end training the whole framework
```bash
python main.py -m fusion_dr -e 20
```


### Testing
You can test on each branch and the whole framework like 
```bash
python main.py -m fusion_dr -l -t
```
Note that we also provided our pretrained model on [Google Drive](https://drive.google.com/open?id=1LxxNY7wbjMXwrRdYJ4hJfXhg9ETAyIuQ)

### Evaluation
The `measurement.py` and `STED` folder is for computation of numerical result mentioned in our paper, including two reconstruction metrics and two decompostion metrics.

### Notes
1. if you train the model on your own dataset(for which topology is different from FaceWarehouse mesh), you have to recompute `Mean_Face.obj` and expression meshes on mean face as mentioned in our paper and regenerate the `FWH_adj_matrix.npz` in `data/disentagle` folder using `src/igl_test.py`.
2. We will release srcipts for data augmentation method metioned in our paper. You could put the augmented interpolated data in `data/disentangle/Interpolated_results`  
3. Currently we have fully tested this package on Ubuntu 16.04 LTS environment with CUDA 9.0. Windows and MacOS are not ensured working.
4. Errors like `Unknown CMake command "pybind11_add_module"` you may encounter while building [get_mesh] and [get_dr] can be solved by 
```bash
git submodule init
git submodule update --recursive
```
5. If you have comments or questions, please contact Zihang Jiang (jzh0103@mail.ustc.edu.cn), Qianyi Wu (wqy9619@mail.ustc.edu.cn), Keyu Chen (cky95@mail.ustc.edu.cn), Juyong Zhang (juyong@ustc.edu.cn).
# Citation
Please cite the following papers if it helps your research: 

<a href="https://arxiv.org/abs/1902.09887">Disentangled Representation Learning for 3D Face Shape</a>

    @inproceedings{Jiang2019Disentangled
          title={Disentangled Representation Learning for 3D Face Shape},
          author={Jiang, Zi-Hang and Wu, Qianyi and Chen, Keyu and Zhang, Juyong}
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
          year={2019},
    }

# Acknowledgement
GCN part code was inspired by https://github.com/tkipf/keras-gcn.
# License
Free for personal or research use, for commercial use please contact us via email.

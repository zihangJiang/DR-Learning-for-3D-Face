# get_dr_py_API
This package is available for using pybind11 to create get_dr python API.

Deformation Representation Feature (DR Feature) is a kind of fundamental and stable tool in geometry learning area. For more background knowledges and properties of DR feature, we suggest these papers in **Citation** section for further reading. 

# Dependencies
This package depends on <a href='https://www.openmesh.org'>OpenMesh</a> and <a href='http://eigen.tuxfamily.org/index.php?title=Main_Page'>Eigen</a> libraries. To run the dr computation code, please make sure you have Eigen and OpenMesh precompiled on your computer and modificate the include path in CMakeList.txt file. 

# Usage
We recommend to use cmake to compile this package and a python API will be automatically created to compute DR features.

After obtaining this repository, run the following code in the repository path:
```
mkdir build
cd build
cmake ..
make -j4
```
You will then get a .so file.
The <a href='https://github.com/QianyiWu/get_dr_py/blob/master/test.py'>test.py</a> with *1_new.obj* and *2_new.obj* could be used for testing.

# Environment Tests
Currently we have fully tested this package on Ubuntu 16.04 LTS environment. Windows and MacOS are not ensured working.

# Citation
Please cite the following papers if it helps your research: 

<a href="https://arxiv.org/abs/1902.09887">Disentangled Representation Learning for 3D Face Shape</a>

    @inproceedings{Jiang2019Disentangled
          title={Disentangled Representation Learning for 3D Face Shape},
          author={Jiang, Zi-Hang and Wu, Qianyi and Chen, Keyu and Zhang, Juyong}
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
          year={2019},
    }

<a href="http://geometrylearning.com/paper/Sparse2018.pdf">Sparse Data Driven Mesh Deformation</a>

    @article{Gao2017SparseDD,
          title={Sparse Data Driven Mesh Deformation},
          author={Lin Gao and Yu-Kun Lai and Jie Yang and Ling-Xiao Zhang and Leif Kobbelt and Shihong Xia},
          journal={CoRR},
          year={2017},
          volume={abs/1709.01250}
    }

<a href="https://arxiv.org/abs/1803.06802v2">Alive Caricature from 2D to 3D</a>

    @inproceedings{wu2018alive,
      title={Alive Caricature from 2D to 3D},
      author={Wu, Qianyi and Zhang, Juyong and Lai, Yu-Kun and Zheng, Jianmin and Cai, Jianfei},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2018},
    }

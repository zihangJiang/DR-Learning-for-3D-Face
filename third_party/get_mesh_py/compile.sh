c++ -O3 -w -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` get_mesh.cpp -o get_mesh`python3-config --extension-suffix` -I/usr/include/eigen3 -I/home/qianyi/get_mesh_py/pybind11 -L/usr/local/lib -lOpenMeshCore -lOpenMeshTools



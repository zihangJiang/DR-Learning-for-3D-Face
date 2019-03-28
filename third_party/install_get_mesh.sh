cd get_mesh_py
mkdir build
cd build
git submodule init
git submodule update --recursive
cmake ..
make -j4
cp get_mesh* ../../../src/

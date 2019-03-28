cd get_dr_py
mkdir build
cd build
git submodule init
git submodule update --recursive
cmake ..
make -j4
cp get_dr* ../../../src/

cmake_minimum_required(VERSION 2.8.12)
project(my_project)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -w -shared -std=c++11")

add_subdirectory("pybind11")

pybind11_add_module(get_dr get_dr.cpp util_3drotation_log_exp.h util_3drotation_log_exp.cpp)


INCLUDE_DIRECTORIES("../eigen-eigen-323c052e1731")
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR})

link_directories("/usr/local/lib")

target_link_libraries(get_dr PRIVATE -lOpenMeshCore -lOpenMeshTools)






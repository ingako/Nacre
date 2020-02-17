cmake_minimum_required(VERSION 3.10)

project(pro_pearl VERSION 0.0.1)

set(sourcefiles
src/pro_pearl_bindings.cpp
src/pro_pearl.cpp
)

set(include_dirs
src/
third_party/
)

set(glog "-lglog -lpthread")
set(warning "-Wall -Wextra -Wno-sign-compare -Wno-nonnull -Wno-unused-variable")
set(flags "-fPIC -std=c++14 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE64 -O3 -DUNIX -g")

add_definitions(${flags})

add_subdirectory(third_party/PEARL)
pybind11_add_module(pro_pearl SHARED ${sourcefiles})

target_link_libraries(pro_pearl PUBLIC pearl)
target_include_directories(pro_pearl PUBLIC ${include_dirs})

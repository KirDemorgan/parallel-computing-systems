cmake_minimum_required(VERSION 3.10)
project(lab2 C)

set(CMAKE_C_STANDARD 99)

# Enable OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

# Add the executable
add_executable(lab2 first/fisrt_sequential.c)

# Link OpenMP to the executable
if(OpenMP_C_FOUND)
    target_link_libraries(lab2 PUBLIC OpenMP::OpenMP_C)
endif()
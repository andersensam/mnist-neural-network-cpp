cmake_minimum_required(VERSION 3.10.0)

project(mnist-neural-network CXX)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -Wextra -Wpedantic -g")

include_directories(../src/include)

add_library(Log ../src/Log.cpp)
add_library(Neural_Network_Layer ../src/Neural_Network_Layer.cpp)
add_library(Neural_Network ../src/Neural_Network.cpp)
add_library(MNIST_Utils ../src/MNIST_Utils.cpp)

target_link_libraries(Neural_Network Neural_Network_Layer)
 
add_executable(mnist-neural-network
    ../src/main.cpp)

target_link_libraries(mnist-neural-network Log)
target_link_libraries(mnist-neural-network Neural_Network)
target_link_libraries(mnist-neural-network MNIST_Utils)
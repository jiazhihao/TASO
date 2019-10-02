cmake_minimum_required(VERSION 3.2)
project(TASO LANGUAGES CXX CUDA)

if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include (${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

#include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories(${CMAKE_BINARY_DIR}) # to include protobuf headeer files
include_directories("include")
include_directories("build")
include_directories("src/core")

#initial variables
set(TASO_LIBS "")
set(TASO_LINK_LIBS ${CMAKE_DL_LIBS})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#Generate protobuf files
find_package(Protobuf REQUIRED)
message(STATUS "PROTOBUF=" ${PROTOBUF_LIBRARY})
include_directories(${PROTOBUF_INCLUDE_DIR})
protobuf_generate_cpp(PROTO_CPPS PROTO_HDRS src/core/rules.proto)
list(APPEND TASO_LINK_LIBS ${PROTOBUF_LIBRARY})

file(GLOB_RECURSE TASO_SRCS
  src/core/*.cc
  )
list(APPEND TASO_SRCS ${PROTO_CPPS})

file(GLOB_RECURSE TASO_CUDA_SRCS
  src/cudnn/*.cu
  )

list(APPEND TASO_SRCS ${TASO_CUDA_SRCS})

#Generic compilation options
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" SUPPORT_CXX11)
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message("Build in Debug mode")
  set(CMAKE_CUDA_FLAGS "-O0 -g -arch compute_70 -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CUDA_FLAGS "-O2 -arch compute_70 -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
endif()

#set CUDA
include(cmake/FindCUDA.cmake)
find_cuda(${USE_CUDA})
if (CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
  message(STATUS "CUDA_INCLUDE_DIR=" ${CUDA_INCLUDE_DIRS})
  add_definitions(-DUSE_CUDNN)
  list(APPEND TASO_LINK_LIBS ${CUDA_CUDART_LIBRARY})
  list(APPEND TASO_LINK_LIBS ${CUDA_CUDA_LIBRARY})
  list(APPEND TASO_LINK_LIBS ${CUDA_CUDNN_LIBRARY})
  list(APPEND TASO_LINK_LIBS ${CUDA_CUBLAS_LIBRARY})
elseif(CUDA_FOUND)
  message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
endif(CUDA_FOUND)

add_library(taso_runtime SHARED ${TASO_SRCS})

set_target_properties(taso_runtime
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(taso_runtime PUBLIC cxx_std_11)

target_link_libraries(taso_runtime ${TASO_LINKER_LIBS})

target_include_directories(taso_runtime
  PUBLIC ${PROJECT_SOURCE_DIR}/include)

set_target_properties(taso_runtime
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#install library
install(TARGETS taso_runtime
    LIBRARY DESTINATION lib)

install (DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION .)

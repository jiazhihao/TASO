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
#include_directories("build")
#include_directories("src/core")

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

file(GLOB_RECURSE TASO_DNNL_SRCS
  src/dnnl/*.cc
  )

#Generic compilation options
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" SUPPORT_CXX11)
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message("Build in Debug mode")
  set(CMAKE_CUDA_FLAGS "-O0 -g -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
endif()

#set CUDA
if (NOT "${USE_CUDA}" STREQUAL "OFF")
  include(cmake/FindCUDA.cmake)
  find_cuda(${USE_CUDA})
  if (CUDA_FOUND)
    list(APPEND TASO_SRCS ${TASO_CUDA_SRCS})
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA_INCLUDE_DIR=" ${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDNN)
    list(APPEND TASO_LINK_LIBS ${CUDA_CUDART_LIBRARY})
    list(APPEND TASO_LINK_LIBS ${CUDA_CUDA_LIBRARY})
    list(APPEND TASO_LINK_LIBS ${CUDA_CUDNN_LIBRARY})
    list(APPEND TASO_LINK_LIBS ${CUDA_CUBLAS_LIBRARY})
  else()
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif(CUDA_FOUND)
endif()

#set DNNL
if (NOT "${USE_DNNL}" STREQUAL "OFF")
  if ("${USE_DNNL}" STREQUAL "ON")
    message(FATAL_ERROR "Auto search DNNL is not supported yet; specify explicit DNNL path")
  elseif(IS_DIRECTORY ${USE_DNNL})
    list(APPEND TASO_SRCS ${TASO_DNNL_SRCS})
    set(DNNL_PATH ${USE_DNNL})
    find_package(OpenMP)
    if (OPENMP_FOUND)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
      message(FATAL_ERROR "OpenMP is not supported")
    endif(OPENMP_FOUND)
    set(DNNL_INCLUDE_DIRS ${DNNL_PATH}/include)
    include_directories(${DNNL_INCLUDE_DIRS})
    message(STATUS "DNNL_INCLUDE_DIRS=" ${DNNL_INCLUDE_DIRS})
    find_library(DNNL_LIBRARY dnnl
      ${DNNL_PATH}/lib)
    message(STATUS "Found DNNL_LIBRARY=" ${DNNL_LIBRARY})
    add_definitions(-DUSE_DNNL)
    list(APPEND TASO_LINK_LIBS ${DNNL_LIBRARY})
  endif()
endif()

add_library(taso_runtime SHARED ${TASO_SRCS})

set_target_properties(taso_runtime
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(taso_runtime PUBLIC cxx_std_11)

target_link_libraries(taso_runtime ${TASO_LINK_LIBS})

target_include_directories(taso_runtime
  PUBLIC ${PROJECT_SOURCE_DIR}/include)

set_target_properties(taso_runtime
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#install library
install(TARGETS taso_runtime
    LIBRARY DESTINATION lib)

install (DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION .)

if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN dnn)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/*.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} taso_runtime)
endif()

cmake_minimum_required(VERSION 3.22)

project(HDMappingCloudVisualCompare LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)

add_subdirectory(${EXTERNAL_DIR}/OpenCL-Headers EXCLUDE_FROM_ALL)
set(ENABLE_OPENCL_LAYERINFO OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/OpenCL-ICD-Loader EXCLUDE_FROM_ALL)

set(ZLIB_COMPAT OFF)
set(WITH_GZFILEOP OFF)
set(BUILD_TESTING OFF)
set(SKIP_INSTALL_ALL ON)
add_subdirectory(${EXTERNAL_DIR}/zlib-ng EXCLUDE_FROM_ALL)

add_library(happly INTERFACE)
target_include_directories(happly SYSTEM INTERFACE ${EXTERNAL_DIR}/happly)

add_library(stb INTERFACE)
target_include_directories(stb SYSTEM INTERFACE ${EXTERNAL_DIR}/stb)

add_library(glm INTERFACE)
target_include_directories(glm SYSTEM INTERFACE ${EXTERNAL_DIR}/glm)

add_library(cli11 INTERFACE)
target_include_directories(cli11 SYSTEM INTERFACE ${EXTERNAL_DIR}/CLI11/include)

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json SYSTEM INTERFACE ${EXTERNAL_DIR}/json/include)

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8 /MP)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(lib)
add_subdirectory(cmd)

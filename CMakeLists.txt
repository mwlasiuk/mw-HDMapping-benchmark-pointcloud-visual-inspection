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

option(CVC_BUILD_GUI "Build the --gui-config view editor (GLFW, OpenGL, Dear ImGui, ImGuizmo)" ON)

# Declared before the warning options below, so third-party sources build without them.
if(CVC_BUILD_GUI)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(${EXTERNAL_DIR}/glfw EXCLUDE_FROM_ALL)

    add_library(glad STATIC ${EXTERNAL_DIR}/glad/src/glad.c)
    target_include_directories(glad SYSTEM PUBLIC ${EXTERNAL_DIR}/glad/include)
    if(WIN32)
        # glad.h includes windows.h, whose min and max macros would break std::min and std::max.
        target_compile_definitions(glad PUBLIC NOMINMAX)
    endif()

    set(IMGUI_DIR ${EXTERNAL_DIR}/imgui)
    add_library(imgui STATIC
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui SYSTEM PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
    target_compile_definitions(imgui PUBLIC IMGUI_DISABLE_OBSOLETE_FUNCTIONS)
    target_link_libraries(imgui PUBLIC glfw)

    add_library(imguizmo STATIC ${EXTERNAL_DIR}/ImGuizmo/src/ImGuizmo.cpp)
    target_include_directories(imguizmo SYSTEM PUBLIC ${EXTERNAL_DIR}/ImGuizmo/src)
    target_link_libraries(imguizmo PUBLIC imgui)
endif()

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8 /MP)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(lib)
if(CVC_BUILD_GUI)
    add_subdirectory(gui)
endif()
add_subdirectory(cmd)

add_library(cvc STATIC
    src/point_cloud.cpp
    src/bounds.cpp
    src/camera.cpp
    src/image.cpp
    src/mosaic.cpp
    src/opencl_renderer.cpp
    src/run_log.cpp
    src/text.cpp
)

target_include_directories(cvc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(cvc PRIVATE CL_TARGET_OPENCL_VERSION=120)

target_link_libraries(cvc
    PUBLIC
        glm
        nlohmann_json
    PRIVATE
        OpenCL::OpenCL
        happly
        stb
        zlib-ng::zlibstatic
)

add_executable(cloud_visual_compare main.cpp)

target_link_libraries(cloud_visual_compare PRIVATE cvc cli11)

if(CVC_BUILD_GUI)
    target_link_libraries(cloud_visual_compare PRIVATE cvc_gui)
    target_compile_definitions(cloud_visual_compare PRIVATE CVC_WITH_GUI)
endif()

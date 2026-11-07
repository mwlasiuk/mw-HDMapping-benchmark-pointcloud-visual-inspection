add_library(cvc_gui STATIC
    src/view_editor.cpp
    src/scene_view.cpp
)

target_include_directories(cvc_gui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(cvc_gui
    PUBLIC
        cvc
    PRIVATE
        glad
        glfw
        imgui
        imguizmo
)

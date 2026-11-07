#include <cvc/gui/view_editor.hpp>

#include "scene_view.hpp"

#include <glad/glad.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>

#include <ImGuizmo.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cvc::gui
{

    namespace
    {

        constexpr std::size_t point_stride       = 10;
        constexpr double      pi                 = 3.14159265358979323846;
        constexpr double      field_of_view      = 45.0 * pi / 180.0;
        constexpr double      infinity           = std::numeric_limits<double>::infinity();
        constexpr float       pick_radius        = 16.0f;
        constexpr double      marker_size_ratio  = 0.05;
        constexpr double      orbit_speed        = 0.006;
        constexpr double      wheel_zoom         = 0.85;
        constexpr double      look_through_frame = 1.08;

        const ImU32 bounds_color         = IM_COL32(110, 160, 220, 200);
        const ImU32 marker_color         = IM_COL32(200, 200, 200, 220);
        const ImU32 edited_color         = IM_COL32(255, 160, 60, 255);
        const ImU32 error_color          = IM_COL32(255, 80, 80, 255);
        const ImU32 selected_color       = IM_COL32(255, 220, 60, 255);
        const ImU32 selected_faint_color = IM_COL32(255, 220, 60, 90);
        const ImU32 selected_fill_color  = IM_COL32(255, 220, 60, 36);
        const ImU32 shadow_color         = IM_COL32(0, 0, 0, 200);

        const ImVec4 warning_text = ImVec4(1.0f, 0.8f, 0.3f, 1.0f);
        const ImVec4 error_text   = ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
        const ImVec4 muted_text   = ImVec4(0.65f, 0.65f, 0.65f, 1.0f);

        std::string& glfw_error()
        {
            static std::string message = "unknown error";
            return message;
        }

        void on_glfw_error(int code, const char* description)
        {
            glfw_error() = std::string(description ? description : "unknown error") + " (" + std::to_string(code) + ")";
        }

        // The window with its OpenGL 3.3 context and Dear ImGui, torn down in reverse order.
        class Window
        {
        public:
            Window(int width, int height, const char* title)
            {
                glfwSetErrorCallback(on_glfw_error);
                if (!glfwInit())
                {
                    throw std::runtime_error("cannot open the view editor, GLFW failed to initialize: " + glfw_error());
                }
                glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
                glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
                glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
                glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
                glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
                glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
                window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
                if (!window_)
                {
                    glfwTerminate();
                    throw std::runtime_error("cannot open the view editor window (OpenGL 3.3 is required): " + glfw_error());
                }
                glfwMakeContextCurrent(window_);
                glfwSwapInterval(1);
                if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
                {
                    glfwDestroyWindow(window_);
                    glfwTerminate();
                    throw std::runtime_error("cannot load the OpenGL 3.3 functions for the view editor");
                }

                IMGUI_CHECKVERSION();
                ImGui::CreateContext();
                ImGui::GetIO().IniFilename = nullptr;
                ImGui::StyleColorsDark();
                scale_            = std::max(1.0f, ImGui_ImplGlfw_GetContentScaleForWindow(window_));
                ImGuiStyle& style = ImGui::GetStyle();
                style.ScaleAllSizes(scale_);
                style.FontScaleDpi = scale_;

                ImGuizmo::Style& gizmo = ImGuizmo::GetStyle();
                gizmo.TranslationLineThickness *= scale_;
                gizmo.TranslationLineArrowSize *= scale_;
                gizmo.RotationLineThickness *= scale_;
                gizmo.RotationOuterLineThickness *= scale_;
                gizmo.ScaleLineThickness *= scale_;
                gizmo.ScaleLineCircleSize *= scale_;
                gizmo.HatchedAxisLineThickness *= scale_;
                gizmo.CenterCircleSize *= scale_;

                ImGui_ImplGlfw_InitForOpenGL(window_, true);
                ImGui_ImplOpenGL3_Init("#version 330 core");
            }

            ~Window()
            {
                ImGui_ImplOpenGL3_Shutdown();
                ImGui_ImplGlfw_Shutdown();
                ImGui::DestroyContext();
                glfwDestroyWindow(window_);
                glfwTerminate();
            }

            Window(const Window&)            = delete;
            Window& operator=(const Window&) = delete;

            GLFWwindow* get() const { return window_; }
            float       scale() const { return scale_; }

        private:
            GLFWwindow* window_ = nullptr;
            float       scale_  = 1.0f;
        };

        // Maps points of the local frame to the screen, clipping lines to the view volume.
        class Projector
        {
        public:
            Projector(const glm::mat4& view_projection, ImVec2 position, ImVec2 size)
                : view_projection_(view_projection), position_(position), size_(size) {}

            bool point(const glm::vec3& p, ImVec2& out) const
            {
                const glm::vec4 c = view_projection_ * glm::vec4(p, 1.0f);
                if (!(c.w > 1e-6f) || c.z < -c.w || c.z > c.w)
                {
                    return false;
                }
                out = screen(c);
                return true;
            }

            void line(ImDrawList* list, const glm::vec3& a, const glm::vec3& b, ImU32 color, float thickness) const
            {
                const glm::vec4 ca = view_projection_ * glm::vec4(a, 1.0f);
                const glm::vec4 cb = view_projection_ * glm::vec4(b, 1.0f);
                float           t0 = 0.0f;
                float           t1 = 1.0f;
                for (int axis = 0; axis < 3; ++axis)
                {
                    for (const float sign : {1.0f, -1.0f})
                    {
                        const float da = ca.w + sign * ca[axis];
                        const float db = cb.w + sign * cb[axis];
                        if (da < 0.0f && db < 0.0f)
                        {
                            return;
                        }
                        if (da < 0.0f)
                        {
                            t0 = std::max(t0, da / (da - db));
                        }
                        else if (db < 0.0f)
                        {
                            t1 = std::min(t1, da / (da - db));
                        }
                    }
                }
                if (t0 < t1)
                {
                    list->AddLine(screen(ca + (cb - ca) * t0), screen(ca + (cb - ca) * t1), color, thickness);
                }
            }

        private:
            ImVec2 screen(const glm::vec4& c) const
            {
                return {position_.x + (0.5f + 0.5f * c.x / c.w) * size_.x, position_.y + (0.5f - 0.5f * c.y / c.w) * size_.y};
            }

            glm::mat4 view_projection_;
            ImVec2    position_;
            ImVec2    size_;
        };

        struct ViewState
        {
            View   view;
            double half_width  = 0.0;
            double half_height = 0.0;
        };

        ViewState state_of(const Shot& shot) { return {shot.view, shot.camera.half_width, shot.camera.half_height}; }

        // Depth range drawn for a view, relative to its target: its clip limits, or the scene bounds where unlimited.
        struct Span
        {
            double front = 0.0;
            double back  = 0.0;

            double center() const { return 0.5 * (front + back); }
            double half() const { return 0.5 * (back - front); }
        };

        Span span_of(const View& view, const Bounds& bounds)
        {
            double lowest  = infinity;
            double highest = -infinity;
            for (const glm::dvec3& corner : bounds.corners())
            {
                const double depth = glm::dot(corner - view.target, view.forward);
                lowest             = std::min(lowest, depth);
                highest            = std::max(highest, depth);
            }
            const double pad = 0.02 * (highest - lowest) + 1e-3;

            Span span{std::isfinite(view.clip_near) ? view.clip_near : lowest - pad,
                      std::isfinite(view.clip_far) ? view.clip_far : highest + pad};
            if (!std::isfinite(view.clip_far))
            {
                span.back = std::max(span.back, span.front + pad);
            }
            if (!std::isfinite(view.clip_near))
            {
                span.front = std::min(span.front, span.back - pad);
            }
            return span;
        }

        struct Angles
        {
            double yaw   = 0.0;
            double pitch = 0.0;
            double roll  = 0.0;
        };

        double degrees(double radians) { return radians * 180.0 / pi; }
        double radians(double degrees) { return degrees * pi / 180.0; }

        // Direction of the view in degrees, with roll measured from the image orientation make_view picks.
        Angles angles_of(const View& view)
        {
            const glm::dvec3& f    = view.forward;
            const View        base = make_view(view.name, view.target, f);
            return {degrees(std::atan2(f.y, f.x)), degrees(std::asin(std::clamp(f.z, -1.0, 1.0))),
                    degrees(std::atan2(glm::dot(view.right, base.up), glm::dot(view.right, base.right)))};
        }

        void set_angles(View& view, const Angles& angles)
        {
            const double     yaw   = radians(angles.yaw);
            const double     pitch = radians(std::clamp(angles.pitch, -90.0, 90.0));
            const double     roll  = radians(angles.roll);
            const glm::dvec3 forward(std::cos(pitch) * std::cos(yaw), std::cos(pitch) * std::sin(yaw), std::sin(pitch));
            const View       base = make_view(view.name, view.target, forward);
            view.forward          = base.forward;
            view.right            = base.right * std::cos(roll) + base.up * std::sin(roll);
            view.up               = glm::cross(view.right, view.forward);
        }

        glm::dvec3 eye_direction(double yaw, double pitch)
        {
            return {std::cos(pitch) * std::cos(yaw), std::cos(pitch) * std::sin(yaw), std::sin(pitch)};
        }

        struct Entry
        {
            Shot        original;
            ViewState   state;
            Shot        shot;
            ShotStatus  status;
            bool        edited = false;
            std::string message;
        };

        enum class Tool
        {
            move,
            rotate,
            resize,
        };

        enum class Drag
        {
            none,
            orbit,
            pan,
        };

        enum class Outcome
        {
            editing,
            finish,
            abort,
        };

        struct Camera
        {
            glm::mat4 view{1.0f};
            glm::mat4 projection{1.0f};
            glm::mat4 view_projection{1.0f};
            bool      orthographic = false;
        };

        class Editor
        {
        public:
            Editor(const EditorScene& scene, const std::vector<Shot>& shots, float scale, std::size_t stride, std::size_t shown)
                : scene_(scene), origin_(scene.reference->origin), scale_(scale), stride_(stride), shown_(shown)
            {
                diagonal_ = std::max(glm::length(scene.bounds.max - scene.bounds.min), 1e-3);
                for (const Shot& shot : shots)
                {
                    Entry entry;
                    entry.original = shot;
                    entries_.push_back(std::move(entry));
                    reset(entries_.back());
                }
                frame_scene();
                orbit_.yaw   = radians(-60.0);
                orbit_.pitch = radians(30.0);
            }

            Outcome frame(bool close_requested)
            {
                bool finish = close_requested;
                bool abort  = false;
                panel(finish, abort);

                const ImGuiIO& io  = ImGui::GetIO();
                viewport_position_ = ImVec2(panel_width_, 0.0f);
                viewport_size_     = ImVec2(std::max(1.0f, io.DisplaySize.x - panel_width_), std::max(1.0f, io.DisplaySize.y));
                camera_            = make_camera_matrices();

                overlay();
                gizmo();
                navigate();
                shortcuts();

                if (abort)
                {
                    return Outcome::abort;
                }
                if (finish)
                {
                    std::string problems;
                    for (const Entry& entry : entries_)
                    {
                        if (!entry.status.error.empty())
                        {
                            problems += entry.shot.view.name + ": " + entry.status.error + "\n";
                        }
                    }
                    if (problems.empty())
                    {
                        return Outcome::finish;
                    }
                    blocked_message_ = "These views cannot be rendered. Fix or reset them, or abort the run.\n\n" + problems;
                    ImGui::OpenPopup("Cannot render yet");
                }
                if (ImGui::BeginPopupModal("Cannot render yet", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
                {
                    ImGui::TextUnformatted(blocked_message_.c_str());
                    if (ImGui::Button("OK", ImVec2(120.0f * scale_, 0.0f)))
                    {
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::EndPopup();
                }
                return Outcome::editing;
            }

            void draw_points(const PointView& points, int framebuffer_height) const
            {
                if (!show_points_)
                {
                    return;
                }
                const ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
                glViewport(static_cast<GLint>(viewport_position_.x * scale.x),
                           static_cast<GLint>(framebuffer_height - (viewport_position_.y + viewport_size_.y) * scale.y),
                           static_cast<GLsizei>(viewport_size_.x * scale.x), static_cast<GLsizei>(viewport_size_.y * scale.y));

                DepthFilter filter;
                const View& view = entries_[selected_].state.view;
                if ((filter_points_ || look_through_) && (std::isfinite(view.clip_near) || std::isfinite(view.clip_far)))
                {
                    constexpr double limit = std::numeric_limits<float>::max();
                    filter.enabled         = true;
                    filter.direction       = glm::vec3(view.forward);
                    filter.offset          = static_cast<float>(glm::dot(view.target - origin_, view.forward));
                    filter.lowest          = static_cast<float>(std::clamp(view.clip_near, -limit, limit));
                    filter.highest         = static_cast<float>(std::clamp(view.clip_far, -limit, limit));
                }

                glEnable(GL_DEPTH_TEST);
                glEnable(GL_PROGRAM_POINT_SIZE);
                points.draw(camera_.view_projection, point_size_ * scale.x, filter);
                glDisable(GL_PROGRAM_POINT_SIZE);
                glDisable(GL_DEPTH_TEST);
            }

            std::vector<Shot> result() const
            {
                std::vector<Shot> shots;
                for (const Entry& entry : entries_)
                {
                    shots.push_back(entry.edited ? entry.shot : entry.original);
                }
                return shots;
            }

        private:
            // --- state of the views ---

            void reset(Entry& entry)
            {
                entry.state  = state_of(entry.original);
                entry.shot   = entry.original;
                entry.edited = false;
                entry.status = scene_.check(entry.shot);
                entry.message.clear();
            }

            void update(Entry& entry)
            {
                entry.edited = true;
                entry.shot   = Shot{entry.state.view, make_camera(entry.state.half_width, entry.state.half_height, scene_.resolution)};
                entry.status = scene_.check(entry.shot);
            }

            glm::dvec3 box_center(const ViewState& state) const
            {
                return state.view.target + state.view.forward * span_of(state.view, scene_.bounds).center();
            }

            // Box of the view in the local frame: columns are right, up and backward scaled by the half extents.
            glm::mat4 box_matrix(const ViewState& state) const
            {
                const Span span = span_of(state.view, scene_.bounds);
                glm::mat4  m(1.0f);
                m[0] = glm::vec4(glm::vec3(state.view.right * state.half_width), 0.0f);
                m[1] = glm::vec4(glm::vec3(state.view.up * state.half_height), 0.0f);
                m[2] = glm::vec4(glm::vec3(-state.view.forward * span.half()), 0.0f);
                m[3] = glm::vec4(glm::vec3(box_center(state) - origin_), 1.0f);
                return m;
            }

            // Reads a box moved, rotated or resized by the gizmo back into the view. A depth limit that was unlimited
            // becomes limited only when its own face is dragged; unlimited sides keep following the scene bounds.
            void apply_box_matrix(ViewState& state, const glm::mat4& m) const
            {
                const glm::dvec3 c0(m[0]);
                const glm::dvec3 c1(m[1]);
                const glm::dvec3 c2(m[2]);
                const double     half_width  = glm::length(c0);
                const double     half_height = glm::length(c1);
                const double     half_depth  = glm::length(c2);
                if (!(half_width > 0.0 && half_height > 0.0 && half_depth > 0.0))
                {
                    return;
                }
                glm::dvec3       forward = -c2 / half_depth;
                glm::dvec3       right   = glm::normalize(c0 - forward * glm::dot(c0, forward));
                glm::dvec3       up      = glm::cross(right, forward);
                const glm::dvec3 center  = glm::dvec3(m[3]) + origin_;

                // The matrix is single precision: keep the exact values the gizmo did not change, so moving a box
                // cannot nudge its pixel size through the rounding in make_camera.
                constexpr double same = 1e-5;
                if (glm::length(forward - state.view.forward) < same && glm::length(right - state.view.right) < same)
                {
                    forward = state.view.forward;
                    right   = state.view.right;
                    up      = state.view.up;
                }
                const auto keep = [](double before, double after)
                { return std::abs(after - before) <= same * before ? before : after; };

                const Span       before       = span_of(state.view, scene_.bounds);
                const glm::dvec3 old_center   = state.view.target + state.view.forward * before.center();
                const glm::dvec3 old_front    = old_center - state.view.forward * before.half();
                const glm::dvec3 old_back     = old_center + state.view.forward * before.half();
                const double     tolerance    = 1e-5 * diagonal_;
                const bool       front_moved  = std::abs(glm::dot(center - forward * half_depth - old_front, forward)) > tolerance;
                const bool       back_moved   = std::abs(glm::dot(center + forward * half_depth - old_back, forward)) > tolerance;
                const bool       near_limited = std::isfinite(state.view.clip_near) || (front_moved && !back_moved);
                const bool       far_limited  = std::isfinite(state.view.clip_far) || (back_moved && !front_moved);

                const double depth   = keep(before.half(), half_depth);
                state.view.forward   = forward;
                state.view.right     = right;
                state.view.up        = up;
                state.view.target    = center;
                state.view.clip_near = near_limited ? -depth : -infinity;
                state.view.clip_far  = far_limited ? depth : infinity;
                state.half_width     = keep(state.half_width, half_width);
                state.half_height    = keep(state.half_height, half_height);
                if (!near_limited && !far_limited)
                {
                    // Without depth limits only the position across the view matters; keep the target on the plane
                    // through the scene center, where the fitted views have it.
                    state.view.target -= forward * glm::dot(center - scene_.bounds.center(), forward);
                }
            }

            // --- side panel ---

            void panel(bool& finish, bool& abort)
            {
                const ImGuiIO& io        = ImGui::GetIO();
                const float    min_width = 300.0f * scale_;
                ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
                ImGui::SetNextWindowSize(ImVec2(440.0f * scale_, io.DisplaySize.y), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowSizeConstraints(ImVec2(min_width, io.DisplaySize.y),
                                                    ImVec2(std::max(min_width, 0.7f * io.DisplaySize.x), io.DisplaySize.y));
                ImGui::Begin("View editor", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
                panel_width_ = ImGui::GetWindowWidth();

                ImGui::PushStyleColor(ImGuiCol_Text, muted_text);
                ImGui::TextWrapped("Close the window or press Render to continue with these views. Views left alone "
                                   "render exactly as fitted.");
                ImGui::PopStyleColor();

                std::size_t edited = 0;
                for (const Entry& entry : entries_)
                {
                    edited += entry.edited ? 1 : 0;
                }
                ImGui::SeparatorText(("Views (" + std::to_string(edited) + " edited)").c_str());
                view_list();

                selected_panel();
                gizmo_panel();
                display_panel();

                ImGui::SeparatorText("Continue");
                const float half = 0.5f * (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x);
                if (ImGui::Button("Render with these views", ImVec2(half, 0.0f)))
                {
                    finish = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Abort run", ImVec2(half, 0.0f)))
                {
                    abort = true;
                }

                ImGui::SeparatorText("Controls");
                ImGui::PushStyleColor(ImGuiCol_Text, muted_text);
                ImGui::TextWrapped("Left drag: orbit. Right or middle drag: pan. Wheel: zoom. Click a camera marker or "
                                   "its name to select it. W, E, R: move, rotate, resize. F: frame the selected view. "
                                   "L: look through it. Home: frame the scene. Page Down, Page Up: next, previous view.");
                ImGui::PopStyleColor();
                ImGui::End();
            }

            void view_list()
            {
                const float rows = static_cast<float>(entries_.size());
                const float height =
                    rows * ImGui::GetTextLineHeightWithSpacing() + 2.0f * ImGui::GetStyle().WindowPadding.y;
                ImGui::BeginChild("views", ImVec2(0.0f, std::min(height, 16.0f * ImGui::GetTextLineHeightWithSpacing())),
                                  ImGuiChildFlags_Borders);
                for (std::size_t i = 0; i < entries_.size(); ++i)
                {
                    const Entry& entry = entries_[i];
                    std::string  label = entry.shot.view.name;
                    if (entry.edited)
                    {
                        label += "  (edited)";
                    }
                    if (!entry.status.error.empty())
                    {
                        label += "  cannot render";
                    }
                    else if (!entry.status.warning.empty())
                    {
                        label += "  warning";
                    }
                    ImGui::PushID(static_cast<int>(i));
                    const bool colored = !entry.status.error.empty() || entry.edited;
                    if (colored)
                    {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(entry.status.error.empty() ? edited_color : error_color));
                    }
                    if (ImGui::Selectable(label.c_str(), selected_ == i))
                    {
                        select(i);
                    }
                    if (colored)
                    {
                        ImGui::PopStyleColor();
                    }
                    ImGui::PopID();
                }
                ImGui::EndChild();
            }

            void selected_panel()
            {
                Entry& entry = entries_[selected_];
                ImGui::SeparatorText(("Selected view " + entry.shot.view.name).c_str());

                ViewState    state   = entry.state;
                bool         changed = false;
                const double speed   = 5e-4 * diagonal_;

                changed |= ImGui::DragScalarN("Center (m)", ImGuiDataType_Double, &state.view.target.x, 3,
                                              static_cast<float>(speed), nullptr, nullptr, "%.3f");

                const Angles angles = angles_of(state.view);
                double       values[3]{angles.yaw, angles.pitch, angles.roll};
                if (ImGui::DragScalarN("Yaw, pitch, roll", ImGuiDataType_Double, values, 3, 0.1f, nullptr, nullptr, "%.2f deg"))
                {
                    set_angles(state.view, {values[0], values[1], values[2]});
                    changed = true;
                }

                double       size[2]{2.0 * state.half_width, 2.0 * state.half_height};
                const double smallest = 2e-3;
                if (ImGui::DragScalarN("Width, height (m)", ImGuiDataType_Double, size, 2, static_cast<float>(speed), &smallest,
                                       nullptr, "%.3f"))
                {
                    state.half_width  = 0.5 * std::max(size[0], smallest);
                    state.half_height = 0.5 * std::max(size[1], smallest);
                    changed           = true;
                }

                const Span span = span_of(state.view, scene_.bounds);
                changed |= depth_limit("Near limit (m)", state.view.clip_near, -infinity, span.front, -infinity,
                                       std::min(state.view.clip_far, span.back), speed);
                changed |= depth_limit("Far limit (m)", state.view.clip_far, infinity, span.back,
                                       std::max(state.view.clip_near, span.front), infinity, speed);
                ImGui::PushStyleColor(ImGuiCol_Text, muted_text);
                ImGui::TextWrapped("Depth limits are measured along the view direction from the center; points outside "
                                   "them are not rendered. Unchecked means unlimited.");
                ImGui::PopStyleColor();

                if (changed)
                {
                    entry.state = state;
                    entry.message.clear();
                    update(entry);
                }

                if (ImGui::Button("Fit to points"))
                {
                    try
                    {
                        const Shot fitted       = scene_.fit(entry.state.view);
                        entry.state.view        = fitted.view;
                        entry.state.half_width  = fitted.camera.half_width;
                        entry.state.half_height = fitted.camera.half_height;
                        update(entry);
                        entry.message = "Fitted the width and height to the reference points within the depth limits.";
                    }
                    catch (const std::exception& e)
                    {
                        entry.message = e.what();
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset"))
                {
                    reset(entry);
                }
                ImGui::SameLine();
                if (ImGui::Button("Frame (F)"))
                {
                    frame_selected();
                }
                ImGui::SameLine();
                if (ImGui::Button(look_through_ ? "Leave look through (L)" : "Look through (L)"))
                {
                    toggle_look_through();
                }

                if (!entry.status.summary.empty())
                {
                    ImGui::TextUnformatted(entry.status.summary.c_str());
                }
                if (!entry.status.warning.empty())
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, warning_text);
                    ImGui::TextWrapped("%s", entry.status.warning.c_str());
                    ImGui::PopStyleColor();
                }
                if (!entry.status.error.empty())
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, error_text);
                    ImGui::TextWrapped("%s", entry.status.error.c_str());
                    ImGui::PopStyleColor();
                }
                if (!entry.message.empty())
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, muted_text);
                    ImGui::TextWrapped("%s", entry.message.c_str());
                    ImGui::PopStyleColor();
                }
            }

            // A checkbox enabling the limit and its value, kept between lowest and highest. Unchecked, the limit is
            // unlimited; checked, it starts where the box face is drawn.
            static bool depth_limit(const char* label, double& limit, double unlimited, double shown, double lowest,
                                    double highest, double speed)
            {
                const double gap     = 1e-3;
                bool         enabled = std::isfinite(limit);
                bool         changed = false;
                ImGui::PushID(label);
                if (ImGui::Checkbox("##enabled", &enabled))
                {
                    limit   = enabled ? shown : unlimited;
                    changed = true;
                }
                ImGui::SameLine();
                ImGui::BeginDisabled(!enabled);
                double value = enabled ? limit : shown;
                if (ImGui::DragScalar(label, ImGuiDataType_Double, &value, static_cast<float>(speed), nullptr, nullptr, "%.3f") && enabled)
                {
                    if (std::isfinite(lowest))
                    {
                        value = std::max(value, lowest + gap);
                    }
                    if (std::isfinite(highest))
                    {
                        value = std::min(value, highest - gap);
                    }
                    limit   = value;
                    changed = true;
                }
                ImGui::EndDisabled();
                ImGui::PopID();
                return changed;
            }

            void gizmo_panel()
            {
                ImGui::SeparatorText("Gizmo");
                int tool = static_cast<int>(tool_);
                ImGui::RadioButton("Move (W)", &tool, static_cast<int>(Tool::move));
                ImGui::SameLine();
                ImGui::RadioButton("Rotate (E)", &tool, static_cast<int>(Tool::rotate));
                ImGui::SameLine();
                ImGui::RadioButton("Resize (R)", &tool, static_cast<int>(Tool::resize));
                tool_ = static_cast<Tool>(tool);

                if (tool_ == Tool::resize)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, muted_text);
                    ImGui::TextWrapped("Drag the handles on the box face turned to you; the opposite face stays in place. "
                                       "Dragging a near or far face sets that depth limit.");
                    ImGui::PopStyleColor();
                    return;
                }
                int mode = mode_ == ImGuizmo::LOCAL ? 0 : 1;
                ImGui::RadioButton("View axes", &mode, 0);
                ImGui::SameLine();
                ImGui::RadioButton("World axes", &mode, 1);
                mode_ = mode == 0 ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
                ImGui::Checkbox("Snap", &snap_);
                if (snap_)
                {
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
                    if (tool_ == Tool::move)
                    {
                        ImGui::DragFloat("m", &move_snap_, 0.01f, 0.001f, 1000.0f, "%.3f");
                    }
                    else
                    {
                        ImGui::DragFloat("deg", &rotate_snap_, 0.1f, 0.1f, 90.0f, "%.1f");
                    }
                }
            }

            void display_panel()
            {
                ImGui::SeparatorText("Display");
                ImGui::Checkbox("Points", &show_points_);
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, muted_text);
                ImGui::Text("every %zuth: %zu of %zu", stride_, shown_, scene_.reference->points.size());
                ImGui::PopStyleColor();
                ImGui::SliderFloat("Point size", &point_size_, 1.0f, 6.0f, "%.0f px");
                ImGui::Checkbox("Only points within the selected depth limits", &filter_points_);
                if (ImGui::Button("Frame scene (Home)"))
                {
                    frame_scene();
                }
            }

            // --- 3D view ---

            Camera make_camera_matrices() const
            {
                const double aspect = viewport_size_.x / viewport_size_.y;
                Camera       camera;
                if (look_through_)
                {
                    const ViewState& state  = entries_[selected_].state;
                    const View&      view   = state.view;
                    const Span       span   = span_of(view, scene_.bounds);
                    const glm::dvec3 center = box_center(state) - origin_ + view.right * look_pan_.x + view.up * look_pan_.y;
                    const double     back   = span.half() + diagonal_;
                    const glm::dvec3 eye    = center - view.forward * back;
                    camera.view             = glm::lookAt(glm::vec3(eye), glm::vec3(eye + view.forward), glm::vec3(view.up));

                    double half_width  = state.half_width * look_through_frame * look_zoom_;
                    double half_height = state.half_height * look_through_frame * look_zoom_;
                    if (half_width / half_height < aspect)
                    {
                        half_width = half_height * aspect;
                    }
                    else
                    {
                        half_height = half_width / aspect;
                    }
                    camera.projection   = glm::ortho(static_cast<float>(-half_width), static_cast<float>(half_width),
                                                     static_cast<float>(-half_height), static_cast<float>(half_height), 0.0f,
                                                     static_cast<float>(2.0 * back + diagonal_));
                    camera.orthographic = true;
                }
                else
                {
                    const glm::dvec3 eye    = orbit_.center + eye_direction(orbit_.yaw, orbit_.pitch) * orbit_.distance;
                    camera.view             = glm::lookAt(glm::vec3(eye), glm::vec3(orbit_.center), glm::vec3(0.0f, 0.0f, 1.0f));
                    const double near_plane = std::max(orbit_.distance * 1e-3, diagonal_ * 1e-7);
                    const double far_plane  = orbit_.distance + 2.0 * diagonal_;
                    camera.projection       = glm::perspective(static_cast<float>(field_of_view), static_cast<float>(aspect),
                                                               static_cast<float>(near_plane), static_cast<float>(far_plane));
                }
                camera.view_projection = camera.projection * camera.view;
                return camera;
            }

            void overlay()
            {
                ImDrawList*     list = ImGui::GetBackgroundDrawList();
                const Projector projector(camera_.view_projection, viewport_position_, viewport_size_);

                const std::array<glm::dvec3, 8> corners = scene_.bounds.corners();
                for (std::size_t a = 0; a < corners.size(); ++a)
                {
                    for (const std::size_t bit : {std::size_t{1}, std::size_t{2}, std::size_t{4}})
                    {
                        if ((a & bit) == 0)
                        {
                            projector.line(list, glm::vec3(corners[a] - origin_), glm::vec3(corners[a | bit] - origin_),
                                           bounds_color, 1.5f * scale_);
                        }
                    }
                }

                markers_.assign(entries_.size(), Marker{});
                for (std::size_t i = 0; i < entries_.size(); ++i)
                {
                    if (i != selected_)
                    {
                        marker(list, projector, i);
                    }
                }
                selected_box(list, projector);
                // Looking through the view, its marker and near face would cover the whole picture.
                if (!look_through_)
                {
                    marker(list, projector, selected_);
                }
            }

            // A small camera glyph at the near face of the view, opening towards the scene.
            void marker(ImDrawList* list, const Projector& projector, std::size_t index)
            {
                const Entry&     entry = entries_[index];
                const View&      view  = entry.state.view;
                const Span       span  = span_of(view, scene_.bounds);
                const double     size  = marker_size_ratio * diagonal_;
                const glm::dvec3 face  = view.target - origin_ + view.forward * span.front;
                const glm::dvec3 apex  = face - view.forward * (1.4 * size);
                const double     ratio = entry.state.half_height / entry.state.half_width;
                const double     bw    = 0.7 * size * (ratio > 1.0 ? 1.0 / ratio : 1.0);
                const double     bh    = bw * ratio;

                const bool  selected  = index == selected_;
                const ImU32 color     = selected ? selected_color : !entry.status.error.empty() ? error_color
                                                                : entry.edited                  ? edited_color
                                                                                                : marker_color;
                const float thickness = (selected ? 2.0f : 1.5f) * scale_;

                const glm::vec3 base[4]{
                    glm::vec3(face - view.right * bw - view.up * bh), glm::vec3(face + view.right * bw - view.up * bh),
                    glm::vec3(face + view.right * bw + view.up * bh), glm::vec3(face - view.right * bw + view.up * bh)};
                for (int k = 0; k < 4; ++k)
                {
                    projector.line(list, glm::vec3(apex), base[k], color, thickness);
                    projector.line(list, base[k], base[(k + 1) % 4], color, thickness);
                }
                const glm::vec3 tip(face + view.up * (bh + 0.6 * bw));
                const glm::vec3 left(face - view.right * (0.5 * bw) + view.up * bh);
                const glm::vec3 right(face + view.right * (0.5 * bw) + view.up * bh);
                projector.line(list, left, tip, color, thickness);
                projector.line(list, tip, right, color, thickness);

                Marker&    placed       = markers_[index];
                const bool apex_visible = projector.point(glm::vec3(apex), placed.apex);
                const bool face_visible = projector.point(glm::vec3(face), placed.face);
                placed.visible          = apex_visible || face_visible;
                if (!apex_visible)
                {
                    placed.apex = placed.face;
                }
                if (!face_visible)
                {
                    placed.face = placed.apex;
                }
                if (placed.visible)
                {
                    const char*  name   = view.name.c_str();
                    const ImVec2 at     = ImVec2(placed.apex.x + 6.0f * scale_, placed.apex.y - ImGui::GetFontSize() - 2.0f * scale_);
                    const ImVec2 extent = ImGui::CalcTextSize(name);
                    list->AddText(ImVec2(at.x + 1.0f, at.y + 1.0f), shadow_color, name);
                    list->AddText(at, color, name);
                    placed.label_min = at;
                    placed.label_max = ImVec2(at.x + extent.x, at.y + extent.y);
                }
            }

            // The whole box the selected view renders: its image rectangle swept over its depth range.
            void selected_box(ImDrawList* list, const Projector& projector) const
            {
                const ViewState& state = entries_[selected_].state;
                const View&      view  = state.view;
                const Span       span  = span_of(view, scene_.bounds);
                glm::vec3        corners[8];
                for (int i = 0; i < 8; ++i)
                {
                    const double u = (i & 1) ? state.half_width : -state.half_width;
                    const double v = (i & 2) ? state.half_height : -state.half_height;
                    const double d = (i & 4) ? span.back : span.front;
                    corners[i]     = glm::vec3(view.target - origin_ + view.right * u + view.up * v + view.forward * d);
                }
                const bool  near_limited = std::isfinite(view.clip_near);
                const bool  far_limited  = std::isfinite(view.clip_far);
                const float thickness    = 2.0f * scale_;
                for (int a = 0; a < 8; ++a)
                {
                    for (const int bit : {1, 2, 4})
                    {
                        if ((a & bit) != 0)
                        {
                            continue;
                        }
                        const int  b     = a | bit;
                        const bool faint = bit != 4 && ((a & 4) ? !far_limited : !near_limited);
                        projector.line(list, corners[a], corners[b], faint ? selected_faint_color : selected_color, thickness);
                    }
                }

                ImVec2 face[4];
                if (!look_through_ && projector.point(corners[0], face[0]) && projector.point(corners[1], face[1]) &&
                    projector.point(corners[3], face[2]) && projector.point(corners[2], face[3]))
                {
                    list->AddQuadFilled(face[0], face[1], face[2], face[3], selected_fill_color);
                }
            }

            void gizmo()
            {
                ImGuizmo::SetOrthographic(camera_.orthographic);
                ImGuizmo::SetRect(viewport_position_.x, viewport_position_.y, viewport_size_.x, viewport_size_.y);

                Entry&       entry  = entries_[selected_];
                glm::mat4    matrix = box_matrix(entry.state);
                const float  unit_box[6]{-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
                const float  move_snap[3]{move_snap_, move_snap_, move_snap_};
                const float  rotate_snap[3]{rotate_snap_, rotate_snap_, rotate_snap_};
                const float* snap = !snap_ ? nullptr : tool_ == Tool::move ? move_snap
                                                   : tool_ == Tool::rotate ? rotate_snap
                                                                           : nullptr;

                ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
                if (tool_ == Tool::rotate)
                {
                    operation = ImGuizmo::ROTATE;
                }
                else if (tool_ == Tool::resize)
                {
                    operation = ImGuizmo::BOUNDS;
                }
                if (ImGuizmo::Manipulate(glm::value_ptr(camera_.view), glm::value_ptr(camera_.projection), operation, mode_,
                                         glm::value_ptr(matrix), nullptr, snap, tool_ == Tool::resize ? unit_box : nullptr))
                {
                    apply_box_matrix(entry.state, matrix);
                    entry.message.clear();
                    update(entry);
                }
            }

            void navigate()
            {
                const ImGuiIO& io     = ImGui::GetIO();
                const ImVec2   mouse  = io.MousePos;
                const bool     inside = mouse.x >= viewport_position_.x && mouse.y >= viewport_position_.y &&
                                    mouse.x < viewport_position_.x + viewport_size_.x &&
                                    mouse.y < viewport_position_.y + viewport_size_.y && !io.WantCaptureMouse;

                if (drag_ == Drag::none && inside)
                {
                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsUsingAny() && !ImGuizmo::IsOver())
                    {
                        drag_    = Drag::orbit;
                        press_   = mouse;
                        dragged_ = false;
                    }
                    else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) || ImGui::IsMouseClicked(ImGuiMouseButton_Middle))
                    {
                        drag_ = Drag::pan;
                    }
                }

                if (drag_ == Drag::orbit)
                {
                    if (ImGuizmo::IsUsingAny())
                    {
                        drag_ = Drag::none;
                    }
                    else if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
                    {
                        const float dx = mouse.x - press_.x;
                        const float dy = mouse.y - press_.y;
                        if (!dragged_ && dx * dx + dy * dy > 9.0f * scale_ * scale_)
                        {
                            dragged_ = true;
                            if (look_through_)
                            {
                                leave_look_through();
                            }
                        }
                        if (dragged_)
                        {
                            orbit_.yaw -= io.MouseDelta.x * orbit_speed;
                            orbit_.pitch = std::clamp(orbit_.pitch + io.MouseDelta.y * orbit_speed, radians(-89.5), radians(89.5));
                        }
                    }
                    else
                    {
                        if (!dragged_)
                        {
                            pick(press_);
                        }
                        drag_ = Drag::none;
                    }
                }
                else if (drag_ == Drag::pan)
                {
                    if (ImGui::IsMouseDown(ImGuiMouseButton_Right) || ImGui::IsMouseDown(ImGuiMouseButton_Middle))
                    {
                        pan(io.MouseDelta);
                    }
                    else
                    {
                        drag_ = Drag::none;
                    }
                }

                if (inside && io.MouseWheel != 0.0f)
                {
                    const double factor = std::pow(wheel_zoom, static_cast<double>(io.MouseWheel));
                    if (look_through_)
                    {
                        look_zoom_ = std::clamp(look_zoom_ * factor, 1e-4, 1e3);
                    }
                    else
                    {
                        orbit_.distance = std::clamp(orbit_.distance * factor, diagonal_ * 1e-5, diagonal_ * 100.0);
                    }
                }
            }

            void pan(const ImVec2& delta)
            {
                if (look_through_)
                {
                    const ViewState& state  = entries_[selected_].state;
                    const double     aspect = viewport_size_.x / viewport_size_.y;
                    double           half   = state.half_height * look_through_frame * look_zoom_;
                    if (state.half_width / state.half_height >= aspect)
                    {
                        half = state.half_width * look_through_frame * look_zoom_ / aspect;
                    }
                    const double per_pixel = 2.0 * half / viewport_size_.y;
                    look_pan_ += glm::dvec2(-delta.x, delta.y) * per_pixel;
                    return;
                }
                const glm::dvec3 forward   = -eye_direction(orbit_.yaw, orbit_.pitch);
                const glm::dvec3 right     = glm::normalize(glm::cross(forward, glm::dvec3(0.0, 0.0, 1.0)));
                const glm::dvec3 up        = glm::cross(right, forward);
                const double     per_pixel = 2.0 * orbit_.distance * std::tan(0.5 * field_of_view) / viewport_size_.y;
                orbit_.center += (-right * static_cast<double>(delta.x) + up * static_cast<double>(delta.y)) * per_pixel;
            }

            void pick(const ImVec2& at)
            {
                const float radius = pick_radius * scale_;
                float       best   = radius * radius;
                std::size_t chosen = selected_;
                for (std::size_t i = 0; i < markers_.size(); ++i)
                {
                    const Marker& candidate = markers_[i];
                    if (!candidate.visible)
                    {
                        continue;
                    }
                    if (at.x >= candidate.label_min.x && at.x <= candidate.label_max.x && at.y >= candidate.label_min.y &&
                        at.y <= candidate.label_max.y)
                    {
                        chosen = i;
                        break;
                    }
                    for (const ImVec2& p : {candidate.apex, candidate.face})
                    {
                        const float dx = p.x - at.x;
                        const float dy = p.y - at.y;
                        if (dx * dx + dy * dy < best)
                        {
                            best   = dx * dx + dy * dy;
                            chosen = i;
                        }
                    }
                }
                select(chosen);
            }

            void shortcuts()
            {
                const ImGuiIO& io = ImGui::GetIO();
                if (io.WantCaptureKeyboard)
                {
                    return;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_W, false))
                {
                    tool_ = Tool::move;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_E, false))
                {
                    tool_ = Tool::rotate;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_R, false))
                {
                    tool_ = Tool::resize;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_F, false))
                {
                    frame_selected();
                }
                if (ImGui::IsKeyPressed(ImGuiKey_L, false))
                {
                    toggle_look_through();
                }
                if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) && look_through_)
                {
                    leave_look_through();
                }
                if (ImGui::IsKeyPressed(ImGuiKey_Home, false))
                {
                    frame_scene();
                }
                // Not Tab: Dear ImGui takes it to move the keyboard focus into the panel.
                const std::size_t count = entries_.size();
                if (ImGui::IsKeyPressed(ImGuiKey_PageDown))
                {
                    select((selected_ + 1) % count);
                }
                if (ImGui::IsKeyPressed(ImGuiKey_PageUp))
                {
                    select((selected_ + count - 1) % count);
                }
            }

            void select(std::size_t index)
            {
                if (index == selected_)
                {
                    return;
                }
                selected_ = index;
                if (look_through_)
                {
                    look_zoom_ = 1.0;
                    look_pan_  = glm::dvec2(0.0);
                }
            }

            void frame_scene()
            {
                if (look_through_)
                {
                    leave_look_through();
                }
                orbit_.center   = scene_.bounds.center() - origin_;
                orbit_.distance = 0.55 * diagonal_ / std::sin(0.5 * field_of_view);
            }

            void frame_selected()
            {
                if (look_through_)
                {
                    look_zoom_ = 1.0;
                    look_pan_  = glm::dvec2(0.0);
                    return;
                }
                const ViewState& state  = entries_[selected_].state;
                const Span       span   = span_of(state.view, scene_.bounds);
                const double     radius = glm::length(glm::dvec3(state.half_width, state.half_height, span.half()));
                orbit_.center           = box_center(state) - origin_;
                orbit_.distance         = std::clamp(1.1 * radius / std::sin(0.5 * field_of_view), diagonal_ * 1e-5, diagonal_ * 100.0);
            }

            void toggle_look_through()
            {
                if (look_through_)
                {
                    leave_look_through();
                    return;
                }
                look_through_ = true;
                look_zoom_    = 1.0;
                look_pan_     = glm::dvec2(0.0);
            }

            // Continues orbiting from where looking through the selected view left the camera.
            void leave_look_through()
            {
                const ViewState& state = entries_[selected_].state;
                const View&      view  = state.view;
                const glm::dvec3 back  = -view.forward;
                const double     half  = std::max(state.half_width, state.half_height) * look_through_frame * look_zoom_;
                orbit_.center          = box_center(state) - origin_ + view.right * look_pan_.x + view.up * look_pan_.y;
                orbit_.yaw             = std::atan2(back.y, back.x);
                orbit_.pitch           = std::clamp(std::asin(std::clamp(back.z, -1.0, 1.0)), radians(-89.5), radians(89.5));
                orbit_.distance        = std::clamp(half / std::tan(0.5 * field_of_view), diagonal_ * 1e-5, diagonal_ * 100.0);
                look_through_          = false;
            }

            struct Orbit
            {
                glm::dvec3 center{0.0};
                double     distance = 1.0;
                double     yaw      = 0.0;
                double     pitch    = 0.0;
            };

            struct Marker
            {
                bool   visible = false;
                ImVec2 apex;
                ImVec2 face;
                ImVec2 label_min;
                ImVec2 label_max;
            };

            const EditorScene& scene_;
            glm::dvec3         origin_;
            double             diagonal_ = 1.0;
            float              scale_    = 1.0f;
            std::size_t        stride_   = 1;
            std::size_t        shown_    = 0;

            std::vector<Entry>  entries_;
            std::size_t         selected_ = 0;
            std::vector<Marker> markers_;

            Orbit      orbit_;
            bool       look_through_ = false;
            double     look_zoom_    = 1.0;
            glm::dvec2 look_pan_{0.0};
            Camera     camera_;
            ImVec2     viewport_position_;
            ImVec2     viewport_size_{1.0f, 1.0f};
            float      panel_width_ = 0.0f;

            Tool           tool_          = Tool::move;
            ImGuizmo::MODE mode_          = ImGuizmo::LOCAL;
            bool           snap_          = false;
            float          move_snap_     = 0.1f;
            float          rotate_snap_   = 5.0f;
            bool           show_points_   = true;
            float          point_size_    = 1.0f;
            bool           filter_points_ = false;
            Drag           drag_          = Drag::none;
            ImVec2         press_;
            bool           dragged_ = false;
            std::string    blocked_message_;
        };

    }

    bool edit_views(const EditorScene& scene, std::vector<Shot>& shots)
    {
        if (scene.reference == nullptr || scene.reference->points.empty() || shots.empty() || !scene.fit || !scene.check ||
            scene.resolution <= 0)
        {
            throw std::invalid_argument("the view editor needs a reference cloud, views, a resolution and callbacks");
        }

        Window window(1600, 1000, "cloud_visual_compare - view editor");

        // Every 10th point; sparser when the GPU cannot hold that many.
        std::optional<PointView> points;
        std::size_t              stride = point_stride;
        while (!points)
        {
            try
            {
                points.emplace(*scene.reference, stride);
            }
            catch (const std::runtime_error&)
            {
                if (scene.reference->points.size() / stride < 1000)
                {
                    throw;
                }
                stride *= 2;
            }
        }

        Editor  editor(scene, shots, window.scale(), stride, points->count());
        Outcome outcome = Outcome::editing;
        while (outcome == Outcome::editing)
        {
            glfwWaitEventsTimeout(0.05);
            if (glfwGetWindowAttrib(window.get(), GLFW_ICONIFIED) != 0 && glfwWindowShouldClose(window.get()) == 0)
            {
                continue;
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGuizmo::BeginFrame();

            const bool close_requested = glfwWindowShouldClose(window.get()) != 0;
            glfwSetWindowShouldClose(window.get(), GLFW_FALSE);
            outcome = editor.frame(close_requested);
            ImGui::Render();

            int width  = 0;
            int height = 0;
            glfwGetFramebufferSize(window.get(), &width, &height);
            glViewport(0, 0, width, height);
            glClearColor(0.08f, 0.08f, 0.09f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            editor.draw_points(*points, height);
            glViewport(0, 0, width, height);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window.get());
        }

        if (outcome == Outcome::abort)
        {
            return false;
        }
        shots = editor.result();
        return true;
    }

}

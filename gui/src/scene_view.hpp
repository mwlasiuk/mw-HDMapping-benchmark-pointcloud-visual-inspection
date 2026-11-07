#pragma once

#include <cvc/point_cloud.hpp>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <cstddef>
#include <limits>

namespace cvc::gui
{

    // Hides points whose depth dot(position, direction) - offset lies outside [lowest, highest].
    struct DepthFilter
    {
        bool      enabled = false;
        glm::vec3 direction{0.0f, 0.0f, 1.0f};
        float     offset  = 0.0f;
        float     lowest  = -std::numeric_limits<float>::max();
        float     highest = std::numeric_limits<float>::max();
    };

    // Every stride-th point of a cloud in a GPU buffer, in the frame of the cloud origin, shaded by intensity.
    // Needs a current OpenGL 3.3 context for its whole lifetime.
    class PointView
    {
    public:
        // Throws when the GPU cannot hold the points.
        PointView(const PointCloud& cloud, std::size_t stride);
        ~PointView();

        PointView(const PointView&)            = delete;
        PointView& operator=(const PointView&) = delete;

        std::size_t count() const { return count_; }

        void draw(const glm::mat4& view_projection, float point_size, const DepthFilter& filter) const;

    private:
        unsigned    program_                  = 0;
        unsigned    array_                    = 0;
        unsigned    buffer_                   = 0;
        int         view_projection_location_ = -1;
        int         point_size_location_      = -1;
        int         filter_plane_location_    = -1;
        int         filter_range_location_    = -1;
        int         filter_enabled_location_  = -1;
        std::size_t count_                    = 0;
    };

}

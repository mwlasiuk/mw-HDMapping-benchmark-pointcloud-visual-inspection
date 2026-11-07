#pragma once

#include <cvc/bounds.hpp>
#include <cvc/point_cloud.hpp>

#include <glm/vec3.hpp>

#include <limits>
#include <string>
#include <vector>

namespace cvc
{

    struct View
    {
        std::string name;
        glm::dvec3  target{0.0};
        glm::dvec3  forward{0.0};
        glm::dvec3  right{0.0};
        glm::dvec3  up{0.0};
        // Depth range along forward measured from target; points outside it are not rendered. Infinite bounds clip nothing.
        double clip_near = -std::numeric_limits<double>::infinity();
        double clip_far  = std::numeric_limits<double>::infinity();
    };

    struct OrthographicCamera
    {
        double half_width  = 0.0;
        double half_height = 0.0;
        int    width       = 0;
        int    height      = 0;
    };

    struct Shot
    {
        View               view;
        OrthographicCamera camera;
    };

    OrthographicCamera make_camera(double half_width, double half_height, int resolution);

    View make_view(const std::string& name, const glm::dvec3& target, const glm::dvec3& forward);

    std::vector<View> make_views(const Bounds& bounds);

    std::vector<Shot> fit_views(const PointCloud& cloud, const std::vector<View>& views, int resolution, double margin,
                                double contained_fraction);

    // Fits the image rectangle of one view to the points within its depth range; orientation and depth range are kept.
    Shot fit_view(const PointCloud& cloud, const View& view, int resolution, double margin, double contained_fraction);

}

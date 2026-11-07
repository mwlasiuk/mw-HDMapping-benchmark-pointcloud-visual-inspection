#include <cvc/camera.hpp>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>

namespace cvc
{

    namespace
    {

        class TrimmedMinimum
        {
        public:
            explicit TrimmedMinimum(std::size_t discard) : keep_(discard + 1) {}

            void add(float value)
            {
                if (!(value < threshold_))
                {
                    return;
                }
                heap_.push(value);
                if (heap_.size() > keep_)
                {
                    heap_.pop();
                }
                if (heap_.size() == keep_)
                {
                    threshold_ = heap_.top();
                }
            }

            float value() const
            {
                if (heap_.empty())
                {
                    throw std::runtime_error("point cloud has no finite coordinates");
                }
                return heap_.top();
            }

        private:
            std::size_t                keep_;
            float                      threshold_ = std::numeric_limits<float>::infinity();
            std::priority_queue<float> heap_;
        };

        struct ProjectedExtent
        {
            explicit ProjectedExtent(std::size_t discard) : low_u(discard), high_u(discard), low_v(discard), high_v(discard) {}

            void add(const glm::vec2& uv)
            {
                low_u.add(uv.x);
                high_u.add(-uv.x);
                low_v.add(uv.y);
                high_v.add(-uv.y);
            }

            glm::dvec2 low() const { return {low_u.value(), low_v.value()}; }
            glm::dvec2 high() const { return {-high_u.value(), -high_v.value()}; }

            TrimmedMinimum low_u;
            TrimmedMinimum high_u;
            TrimmedMinimum low_v;
            TrimmedMinimum high_v;
        };

        void check_fit_arguments(const PointCloud& cloud, int resolution, double contained_fraction)
        {
            if (resolution <= 0)
            {
                throw std::invalid_argument("camera resolution must be positive");
            }
            if (!(contained_fraction > 0.0 && contained_fraction <= 1.0))
            {
                throw std::invalid_argument("contained fraction must be in (0, 1]");
            }
            if (cloud.points.empty())
            {
                throw std::invalid_argument("cannot fit views to an empty point cloud");
            }
        }

        std::size_t discard_count(std::size_t points, double contained_fraction)
        {
            return static_cast<std::size_t>(std::floor(static_cast<double>(points) * (1.0 - contained_fraction) / 4.0));
        }

        // Centers the view on the extent of the cloud projected onto its image plane, with margin around it.
        Shot make_shot(const PointCloud& cloud, const View& view, const ProjectedExtent& extent, int resolution, double margin)
        {
            const glm::dvec3 offset = cloud.origin - view.target;
            const glm::dvec2 shift(glm::dot(offset, view.right), glm::dot(offset, view.up));
            const glm::dvec2 low    = extent.low() + shift;
            const glm::dvec2 high   = extent.high() + shift;
            const glm::dvec2 center = 0.5 * (low + high);
            const glm::dvec2 half   = glm::max(0.5 * (high - low) * margin, glm::dvec2(1e-6));

            Shot shot{view, make_camera(half.x, half.y, resolution)};
            shot.view.target = view.target + view.right * center.x + view.up * center.y;
            return shot;
        }

    }

    OrthographicCamera make_camera(double half_width, double half_height, int resolution)
    {
        if (resolution <= 0)
        {
            throw std::invalid_argument("camera resolution must be positive");
        }

        OrthographicCamera camera;
        if (half_width >= half_height)
        {
            camera.width       = resolution;
            camera.height      = std::max(1, static_cast<int>(std::ceil(resolution * half_height / half_width)));
            camera.half_width  = half_width;
            camera.half_height = half_width * camera.height / camera.width;
        }
        else
        {
            camera.height      = resolution;
            camera.width       = std::max(1, static_cast<int>(std::ceil(resolution * half_width / half_height)));
            camera.half_height = half_height;
            camera.half_width  = half_height * camera.width / camera.height;
        }
        return camera;
    }

    View make_view(const std::string& name, const glm::dvec3& target, const glm::dvec3& forward)
    {
        const glm::dvec3 f        = glm::normalize(forward);
        const glm::dvec3 world_up = std::abs(f.z) > 0.999 ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(0.0, 0.0, 1.0);
        const glm::dvec3 right    = glm::normalize(glm::cross(f, world_up));
        const glm::dvec3 up       = glm::cross(right, f);
        return {name, target, f, right, up};
    }

    std::vector<View> make_views(const Bounds& bounds)
    {
        const glm::dvec3 center   = bounds.center();
        const glm::dvec3 half     = bounds.half_extent();
        const glm::dvec3 diagonal = glm::max(half, glm::dvec3(std::max(glm::length(half) * 1e-3, 1e-9)));

        struct Direction
        {
            const char* name;
            glm::dvec3  position;
        };
        const Direction axes[] = {
            {"px", {1.0, 0.0, 0.0}},
            {"nx", {-1.0, 0.0, 0.0}},
            {"py", {0.0, 1.0, 0.0}},
            {"ny", {0.0, -1.0, 0.0}},
            {"pz", {0.0, 0.0, 1.0}},
            {"nz", {0.0, 0.0, -1.0}},
        };

        std::vector<View> views;
        views.reserve(14);
        for (const Direction& axis : axes)
        {
            views.push_back(make_view(axis.name, center, -axis.position));
        }

        for (int i = 0; i < 8; ++i)
        {
            const bool        px = (i & 1) != 0;
            const bool        py = (i & 2) != 0;
            const bool        pz = (i & 4) != 0;
            const glm::dvec3  sign(px ? 1.0 : -1.0, py ? 1.0 : -1.0, pz ? 1.0 : -1.0);
            const std::string name = std::string(px ? "px" : "nx") + "_" + (py ? "py" : "ny") + "_" + (pz ? "pz" : "nz");
            views.push_back(make_view(name, center, -sign * diagonal));
        }

        return views;
    }

    std::vector<Shot> fit_views(const PointCloud& cloud, const std::vector<View>& views, int resolution, double margin,
                                double contained_fraction)
    {
        check_fit_arguments(cloud, resolution, contained_fraction);
        const std::size_t discard = discard_count(cloud.points.size(), contained_fraction);

        const std::size_t      count = views.size();
        std::vector<glm::vec3> rights(count);
        std::vector<glm::vec3> ups(count);
        for (std::size_t k = 0; k < count; ++k)
        {
            rights[k] = glm::vec3(views[k].right);
            ups[k]    = glm::vec3(views[k].up);
        }

        std::vector<ProjectedExtent> extents(count, ProjectedExtent(discard));
        for (const Point& p : cloud.points)
        {
            for (std::size_t k = 0; k < count; ++k)
            {
                extents[k].add(glm::vec2(glm::dot(p.position, rights[k]), glm::dot(p.position, ups[k])));
            }
        }

        std::vector<Shot> shots;
        shots.reserve(count);
        for (std::size_t k = 0; k < count; ++k)
        {
            shots.push_back(make_shot(cloud, views[k], extents[k], resolution, margin));
        }
        return shots;
    }

    Shot fit_view(const PointCloud& cloud, const View& view, int resolution, double margin, double contained_fraction)
    {
        check_fit_arguments(cloud, resolution, contained_fraction);

        // Depth of a point relative to the target, computed in the frame of the cloud origin like the renderer does.
        const glm::vec3 forward(view.forward);
        const glm::vec3 right(view.right);
        const glm::vec3 up(view.up);
        const auto      offset = static_cast<float>(glm::dot(view.target - cloud.origin, view.forward));
        const auto      inside = [&](const Point& p)
        {
            const float z = glm::dot(p.position, forward) - offset;
            return !(z < view.clip_near || z > view.clip_far);
        };

        const auto contained = static_cast<std::size_t>(std::count_if(cloud.points.begin(), cloud.points.end(), inside));
        if (contained == 0)
        {
            throw std::runtime_error("no points of the cloud lie within the depth range of view " + view.name);
        }

        ProjectedExtent extent(discard_count(contained, contained_fraction));
        for (const Point& p : cloud.points)
        {
            if (inside(p))
            {
                extent.add(glm::vec2(glm::dot(p.position, right), glm::dot(p.position, up)));
            }
        }
        return make_shot(cloud, view, extent, resolution, margin);
    }

}

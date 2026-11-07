#pragma once

#include <cvc/camera.hpp>
#include <cvc/mosaic.hpp>

#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>

#include <cmath>
#include <limits>

// JSON conversions of the view, camera and mosaic types. Doubles are written with enough digits to read back exactly.

namespace nlohmann
{

    template <>
    struct adl_serializer<glm::dvec3>
    {
        template <typename Json>
        static void to_json(Json& json, const glm::dvec3& value)
        {
            json = Json::array({value.x, value.y, value.z});
        }

        template <typename Json>
        static void from_json(const Json& json, glm::dvec3& value)
        {
            value = {json.at(0).template get<double>(), json.at(1).template get<double>(), json.at(2).template get<double>()};
        }
    };

}

namespace cvc
{

    // JSON has no infinity (nlohmann writes it as null, which then fails to read back as a number), so an unbounded
    // depth limit is left out and a missing one reads back as unbounded. Views without clipping keep the plain layout.
    template <typename Json, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<Json>::value, int> = 0>
    void to_json(Json& json, const View& view)
    {
        json            = Json::object();
        json["name"]    = view.name;
        json["target"]  = view.target;
        json["forward"] = view.forward;
        json["right"]   = view.right;
        json["up"]      = view.up;
        if (std::isfinite(view.clip_near))
        {
            json["clip_near"] = view.clip_near;
        }
        if (std::isfinite(view.clip_far))
        {
            json["clip_far"] = view.clip_far;
        }
    }

    template <typename Json, nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<Json>::value, int> = 0>
    void from_json(const Json& json, View& view)
    {
        json.at("name").get_to(view.name);
        json.at("target").get_to(view.target);
        json.at("forward").get_to(view.forward);
        json.at("right").get_to(view.right);
        json.at("up").get_to(view.up);
        view.clip_near = json.contains("clip_near") ? json.at("clip_near").template get<double>() : -std::numeric_limits<double>::infinity();
        view.clip_far  = json.contains("clip_far") ? json.at("clip_far").template get<double>() : std::numeric_limits<double>::infinity();
    }

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(OrthographicCamera, half_width, half_height, width, height)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Shot, view, camera)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MosaicLayout, columns, rows, tile_width, tile_height, gap)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MosaicPlan, tile_camera, layout, tile_channels, channels)

}

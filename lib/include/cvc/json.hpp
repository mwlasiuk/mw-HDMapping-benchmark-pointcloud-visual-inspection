#pragma once

#include <cvc/camera.hpp>
#include <cvc/mosaic.hpp>

#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>

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

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(View, name, target, forward, right, up)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(OrthographicCamera, half_width, half_height, width, height)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Shot, view, camera)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MosaicLayout, columns, rows, tile_width, tile_height, gap)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MosaicPlan, tile_camera, layout, tile_channels, channels)

}

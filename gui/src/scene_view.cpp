#include "scene_view.hpp"

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cvc::gui
{

    namespace
    {

        const char* const vertex_source = R"GLSL(
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in float shade;
uniform mat4 view_projection;
uniform float point_size;
uniform vec4 filter_plane;
uniform vec2 filter_range;
uniform int filter_enabled;
out float vertex_shade;
void main()
{
    float depth = dot(position, filter_plane.xyz) - filter_plane.w;
    if (filter_enabled != 0 && (depth < filter_range.x || depth > filter_range.y)) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    } else {
        gl_Position = view_projection * vec4(position, 1.0);
    }
    gl_PointSize = point_size;
    vertex_shade = shade;
}
)GLSL";

        const char* const fragment_source = R"GLSL(
#version 330 core
in float vertex_shade;
out vec4 color;
void main()
{
    color = vec4(vec3(mix(0.18, 1.0, vertex_shade)), 1.0);
}
)GLSL";

        struct Vertex
        {
            float        x;
            float        y;
            float        z;
            std::uint8_t shade;
            std::uint8_t padding[3];
        };

        static_assert(sizeof(Vertex) == 16);

        GLuint compile(GLenum type, const char* source)
        {
            const GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &source, nullptr);
            glCompileShader(shader);
            GLint ok = GL_FALSE;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
            if (ok != GL_TRUE)
            {
                GLint size = 0;
                glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &size);
                std::string log(static_cast<std::size_t>(std::max(size, 1)), '\0');
                glGetShaderInfoLog(shader, size, nullptr, log.data());
                glDeleteShader(shader);
                throw std::runtime_error("OpenGL shader compilation failed:\n" + log);
            }
            return shader;
        }

        GLuint link(const char* vertex, const char* fragment)
        {
            const GLuint vertex_shader   = compile(GL_VERTEX_SHADER, vertex);
            const GLuint fragment_shader = compile(GL_FRAGMENT_SHADER, fragment);
            const GLuint program         = glCreateProgram();
            glAttachShader(program, vertex_shader);
            glAttachShader(program, fragment_shader);
            glLinkProgram(program);
            glDeleteShader(vertex_shader);
            glDeleteShader(fragment_shader);
            GLint ok = GL_FALSE;
            glGetProgramiv(program, GL_LINK_STATUS, &ok);
            if (ok != GL_TRUE)
            {
                GLint size = 0;
                glGetProgramiv(program, GL_INFO_LOG_LENGTH, &size);
                std::string log(static_cast<std::size_t>(std::max(size, 1)), '\0');
                glGetProgramInfoLog(program, size, nullptr, log.data());
                glDeleteProgram(program);
                throw std::runtime_error("OpenGL program link failed:\n" + log);
            }
            return program;
        }

        // Intensity quantile of the sampled points; reorders values.
        float quantile(std::vector<float>& values, double q)
        {
            const auto k = static_cast<std::ptrdiff_t>(q * static_cast<double>(values.size() - 1));
            std::nth_element(values.begin(), values.begin() + k, values.end());
            return values[static_cast<std::size_t>(k)];
        }

    }

    PointView::PointView(const PointCloud& cloud, std::size_t stride)
    {
        if (stride == 0)
        {
            throw std::invalid_argument("point stride must be positive");
        }
        count_ = (cloud.points.size() + stride - 1) / stride;

        // Without an intensity range every point is drawn bright, as the renderer does.
        float low   = -1.0f;
        float scale = 0.0f;
        if (count_ > 0)
        {
            std::vector<float> intensities(count_);
            for (std::size_t i = 0; i < count_; ++i)
            {
                intensities[i] = cloud.points[i * stride].intensity;
            }
            low              = quantile(intensities, 0.01);
            const float high = quantile(intensities, 0.99);
            scale            = high > low ? 255.0f / (high - low) : 0.0f;
        }

        program_                  = link(vertex_source, fragment_source);
        view_projection_location_ = glGetUniformLocation(program_, "view_projection");
        point_size_location_      = glGetUniformLocation(program_, "point_size");
        filter_plane_location_    = glGetUniformLocation(program_, "filter_plane");
        filter_range_location_    = glGetUniformLocation(program_, "filter_range");
        filter_enabled_location_  = glGetUniformLocation(program_, "filter_enabled");

        glGenVertexArrays(1, &array_);
        glGenBuffers(1, &buffer_);
        glBindVertexArray(array_);
        glBindBuffer(GL_ARRAY_BUFFER, buffer_);

        // Written straight into the mapped buffer, so no copy of the sampled points is kept in RAM.
        const auto bytes = static_cast<GLsizeiptr>(std::max<std::size_t>(count_, 1) * sizeof(Vertex));
        while (glGetError() != GL_NO_ERROR)
        {
        }
        glBufferData(GL_ARRAY_BUFFER, bytes, nullptr, GL_STATIC_DRAW);
        void* mapped = glGetError() == GL_NO_ERROR ? glMapBufferRange(GL_ARRAY_BUFFER, 0, bytes, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT) : nullptr;
        if (mapped == nullptr)
        {
            glBindVertexArray(0);
            glDeleteBuffers(1, &buffer_);
            glDeleteVertexArrays(1, &array_);
            glDeleteProgram(program_);
            throw std::runtime_error("cannot allocate " + std::to_string(bytes) + " bytes of GPU memory for " +
                                     std::to_string(count_) + " points");
        }

        auto* vertices = static_cast<Vertex*>(mapped);
        for (std::size_t i = 0; i < count_; ++i)
        {
            const Point& p = cloud.points[i * stride];
            const float  t = scale > 0.0f ? std::clamp((p.intensity - low) * scale, 0.0f, 255.0f) : 255.0f;
            vertices[i]    = {p.position.x, p.position.y, p.position.z, static_cast<std::uint8_t>(t + 0.5f), {0, 0, 0}};
        }
        glUnmapBuffer(GL_ARRAY_BUFFER);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void*>(offsetof(Vertex, x)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 1, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), reinterpret_cast<const void*>(offsetof(Vertex, shade)));
        glBindVertexArray(0);
    }

    PointView::~PointView()
    {
        glDeleteBuffers(1, &buffer_);
        glDeleteVertexArrays(1, &array_);
        glDeleteProgram(program_);
    }

    void PointView::draw(const glm::mat4& view_projection, float point_size, const DepthFilter& filter) const
    {
        if (count_ == 0)
        {
            return;
        }
        glUseProgram(program_);
        glUniformMatrix4fv(view_projection_location_, 1, GL_FALSE, glm::value_ptr(view_projection));
        glUniform1f(point_size_location_, point_size);
        glUniform4f(filter_plane_location_, filter.direction.x, filter.direction.y, filter.direction.z, filter.offset);
        glUniform2f(filter_range_location_, filter.lowest, filter.highest);
        glUniform1i(filter_enabled_location_, filter.enabled ? 1 : 0);
        glBindVertexArray(array_);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(count_));
        glBindVertexArray(0);
        glUseProgram(0);
    }

}

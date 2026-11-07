#include <cvc/opencl_renderer.hpp>

#include <CL/cl.h>
#include <glm/vec3.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cvc
{

    namespace
    {

        const char* const kernel_source = R"CLC(
uint encode_depth(float depth)
{
    uint bits = as_uint(depth);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

bool project(float4 point, float4 target, float4 right, float4 up, float4 forward, float2 scale, float2 clip,
             int width, int height, int radius, int* px, int* py, uint* key)
{
    float3 d = point.xyz - target.xyz;
    float u = dot(d, right.xyz) * scale.x + 0.5f * (float)width;
    float v = 0.5f * (float)height - dot(d, up.xyz) * scale.y;
    if (!(u >= -(float)radius && u < (float)(width + radius) && v >= -(float)radius && v < (float)(height + radius))) {
        return false;
    }
    float z = dot(d, forward.xyz);
    if (z < clip.x || z > clip.y) {
        return false;
    }
    *px = (int)floor(u);
    *py = (int)floor(v);
    *key = encode_depth(z);
    return true;
}

__kernel void splat_depth(__global const float4* points, uint count,
                          float4 target, float4 right, float4 up, float4 forward, float2 scale, float2 clip,
                          int width, int height, int radius,
                          __global uint* depth)
{
    uint i = get_global_id(0);
    if (i >= count) {
        return;
    }
    int px, py;
    uint key;
    if (!project(points[i], target, right, up, forward, scale, clip, width, height, radius, &px, &py, &key)) {
        return;
    }
    int x0 = max(px - radius, 0);
    int x1 = min(px + radius, width - 1);
    int y0 = max(py - radius, 0);
    int y1 = min(py + radius, height - 1);
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            size_t p = (size_t)y * (size_t)width + (size_t)x;
            if (key < depth[p]) {
                atomic_min(&depth[p], key);
            }
        }
    }
}

__kernel void splat_index(__global const float4* points, uint count,
                          float4 target, float4 right, float4 up, float4 forward, float2 scale, float2 clip,
                          int width, int height, int radius,
                          __global const uint* depth, __global uint* index)
{
    uint i = get_global_id(0);
    if (i >= count) {
        return;
    }
    int px, py;
    uint key;
    if (!project(points[i], target, right, up, forward, scale, clip, width, height, radius, &px, &py, &key)) {
        return;
    }
    int x0 = max(px - radius, 0);
    int x1 = min(px + radius, width - 1);
    int y0 = max(py - radius, 0);
    int y1 = min(py + radius, height - 1);
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            size_t p = (size_t)y * (size_t)width + (size_t)x;
            if (depth[p] == key) {
                atomic_min(&index[p], i);
            }
        }
    }
}

__constant float3 cividis_stops[10] = {
    (float3)(0.0000f, 0.1333f, 0.3059f), (float3)(0.0706f, 0.2078f, 0.4392f), (float3)(0.2314f, 0.2863f, 0.4235f),
    (float3)(0.3412f, 0.3647f, 0.4275f), (float3)(0.4392f, 0.4431f, 0.4510f), (float3)(0.5412f, 0.5255f, 0.4706f),
    (float3)(0.6471f, 0.6118f, 0.4549f), (float3)(0.7647f, 0.7020f, 0.4118f), (float3)(0.8824f, 0.8000f, 0.3333f),
    (float3)(0.9961f, 0.9098f, 0.2196f)};

__constant float3 blues_stops[9] = {
    (float3)(0.9686f, 0.9843f, 1.0000f), (float3)(0.8706f, 0.9216f, 0.9686f), (float3)(0.7765f, 0.8588f, 0.9373f),
    (float3)(0.6196f, 0.7922f, 0.8824f), (float3)(0.4196f, 0.6824f, 0.8392f), (float3)(0.2588f, 0.5725f, 0.7765f),
    (float3)(0.1294f, 0.4431f, 0.7098f), (float3)(0.0314f, 0.3176f, 0.6118f), (float3)(0.0314f, 0.1882f, 0.4196f)};

float3 polynomial(float t, float3 c0, float3 c1, float3 c2, float3 c3, float3 c4, float3 c5, float3 c6)
{
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

float3 viridis(float t)
{
    return polynomial(t,
                      (float3)(0.2777273272234177f, 0.005407344544966578f, 0.3340998053353061f),
                      (float3)(0.1050930431085774f, 1.404613529898575f, 1.384590162594685f),
                      (float3)(-0.3308618287255563f, 0.214847559468213f, 0.09509516302823659f),
                      (float3)(-4.634230498983486f, -5.799100973351585f, -19.33244095627987f),
                      (float3)(6.228269936347081f, 14.17993336680509f, 56.69055260068105f),
                      (float3)(4.776384997670288f, -13.74514537774601f, -65.35303263337234f),
                      (float3)(-5.435455855934631f, 4.645852612178535f, 26.3124352495832f));
}

float3 turbo(float t)
{
    float4 red_low    = (float4)(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f);
    float4 green_low  = (float4)(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f);
    float4 blue_low   = (float4)(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f);
    float2 red_high   = (float2)(-152.94239396f, 59.28637943f);
    float2 green_high = (float2)(4.27729857f, 2.82956604f);
    float2 blue_high  = (float2)(-89.90310912f, 27.34824973f);
    float4 low        = (float4)(1.0f, t, t * t, t * t * t);
    float2 high       = low.zw * low.z;
    return (float3)(dot(low, red_low) + dot(high, red_high),
                    dot(low, green_low) + dot(high, green_high),
                    dot(low, blue_low) + dot(high, blue_high));
}

float3 inferno(float t)
{
    return polynomial(t,
                      (float3)(0.0002189403691192265f, 0.001651004631001012f, -0.01948089843709184f),
                      (float3)(0.1065134194856116f, 0.5639564367884091f, 3.932712388889277f),
                      (float3)(11.60249308247187f, -3.972853965665698f, -15.9423941062914f),
                      (float3)(-41.70399613139459f, 17.43639888205313f, 44.35414519872813f),
                      (float3)(77.162935699427f, -33.40235894210092f, -81.80730925738993f),
                      (float3)(-71.31942824499214f, 32.62606426397723f, 73.20951985803202f),
                      (float3)(25.13112622477341f, -12.24266895238567f, -23.07032500287172f));
}

float3 plasma(float t)
{
    return polynomial(t,
                      (float3)(0.05873234392399702f, 0.02333670892565664f, 0.5433401826748754f),
                      (float3)(2.176514634195958f, 0.2383834171260182f, 0.7539604599784036f),
                      (float3)(-2.689460476458034f, -7.455851135738909f, 3.110799939717086f),
                      (float3)(6.130348345893603f, 42.3461881477227f, -28.51885465332158f),
                      (float3)(-11.10743619062271f, -82.66631109428045f, 60.13984767418263f),
                      (float3)(10.02306557647065f, 71.41361770095349f, -54.07218655560067f),
                      (float3)(-3.658713842777788f, -22.93153465461149f, 18.19190778539828f));
}

float3 cividis(float t)
{
    float scaled = t * 9.0f;
    int   index  = min((int)scaled, 8);
    return mix(cividis_stops[index], cividis_stops[index + 1], scaled - (float)index);
}

float3 magma(float t)
{
    return polynomial(t,
                      (float3)(-0.002136485053939582f, -0.000749655052795221f, -0.005386127855323933f),
                      (float3)(0.2516605407371642f, 0.6775232436837668f, 2.494026599312351f),
                      (float3)(8.353717279216625f, -3.577719514958484f, 0.3144679030132573f),
                      (float3)(-27.66873308576866f, 14.26473078096533f, -13.64921318813922f),
                      (float3)(52.17613981234068f, -27.94360607168351f, 12.94416944238394f),
                      (float3)(-50.76852536473588f, 29.04658282127291f, 4.23415299384598f),
                      (float3)(18.65570506591883f, -11.48977351997711f, -5.601961508734096f));
}

float3 cubehelix(float t)
{
    float amplitude = t * (1.0f - t) * 0.5f;
    float angle     = 6.283185307179586f * (0.5f / 3.0f - 1.5f * t);
    float cosine    = cos(angle);
    float sine      = sin(angle);
    return t + amplitude * (float3)(-0.14861f * cosine + 1.78277f * sine,
                                    -0.29227f * cosine - 0.90649f * sine,
                                    1.97294f * cosine);
}

float3 hot(float t)
{
    float3 ramp_begin = (float3)(0.0f, 0.365079f, 0.746032f);
    float3 ramp_end   = (float3)(0.365079f, 0.746032f, 1.0f);
    float3 color      = clamp(((float3)(t) - ramp_begin) / (ramp_end - ramp_begin), 0.0f, 1.0f);
    color.x           = mix(0.0416f, 1.0f, color.x);
    return color;
}

float3 blues(float t)
{
    float scaled = t * 8.0f;
    int   index  = min((int)scaled, 7);
    return mix(blues_stops[index], blues_stops[index + 1], scaled - (float)index);
}

float3 blue_white_red(float t)
{
    if (t < 0.5f) {
        return mix((float3)(0.0f, 0.0f, 1.0f), (float3)(1.0f), t * 2.0f);
    }
    return mix((float3)(1.0f), (float3)(1.0f, 0.0f, 0.0f), t * 2.0f - 1.0f);
}

float3 apply_colormap(int colormap, float t)
{
    switch (colormap) {
    case 1:
        return turbo(t);
    case 2:
        return inferno(t);
    case 3:
        return plasma(t);
    case 4:
        return cividis(t);
    case 5:
        return magma(t);
    case 6:
        return cubehelix(t);
    case 7:
        return hot(t);
    case 8:
        return blues(t);
    case 9:
        return blue_white_red(t);
    default:
        return viridis(t);
    }
}

__kernel void resolve(__global const uint* index, __global const float4* points, uint pixel_count,
                      float intensity_min, float intensity_scale, int channels, int colormap,
                      __global uchar* color)
{
    uint i = get_global_id(0);
    if (i >= pixel_count) {
        return;
    }
    __global uchar* pixel = color + (size_t)i * (size_t)channels;
    uint id = index[i];
    if (id == 0xFFFFFFFFu) {
        for (int c = 0; c < channels; ++c) {
            pixel[c] = 0;
        }
        return;
    }
    float t = intensity_scale > 0.0f ? clamp((points[id].w - intensity_min) * intensity_scale, 0.0f, 1.0f) : 1.0f;
    if (channels <= 2) {
        pixel[0] = convert_uchar_sat_rte(t * 255.0f);
    }
    else {
        float3 rgb = clamp(apply_colormap(colormap, t), 0.0f, 1.0f);
        pixel[0] = convert_uchar_sat_rte(rgb.x * 255.0f);
        pixel[1] = convert_uchar_sat_rte(rgb.y * 255.0f);
        pixel[2] = convert_uchar_sat_rte(rgb.z * 255.0f);
    }
    if (channels == 2 || channels == 4) {
        pixel[channels - 1] = (uchar)255;
    }
}
)CLC";

        struct ReleaseContext
        {
            void operator()(cl_context handle) const { clReleaseContext(handle); }
        };
        struct ReleaseQueue
        {
            void operator()(cl_command_queue handle) const { clReleaseCommandQueue(handle); }
        };
        struct ReleaseProgram
        {
            void operator()(cl_program handle) const { clReleaseProgram(handle); }
        };
        struct ReleaseKernel
        {
            void operator()(cl_kernel handle) const { clReleaseKernel(handle); }
        };
        struct ReleaseMem
        {
            void operator()(cl_mem handle) const { clReleaseMemObject(handle); }
        };

        using ContextHandle = std::unique_ptr<std::remove_pointer_t<cl_context>, ReleaseContext>;
        using QueueHandle   = std::unique_ptr<std::remove_pointer_t<cl_command_queue>, ReleaseQueue>;
        using ProgramHandle = std::unique_ptr<std::remove_pointer_t<cl_program>, ReleaseProgram>;
        using KernelHandle  = std::unique_ptr<std::remove_pointer_t<cl_kernel>, ReleaseKernel>;
        using MemHandle     = std::unique_ptr<std::remove_pointer_t<cl_mem>, ReleaseMem>;

        constexpr cl_uint invalid_index = 0xFFFFFFFFu;

        void check(cl_int status, const char* what)
        {
            if (status != CL_SUCCESS)
            {
                throw std::runtime_error(std::string("OpenCL ") + what + " failed with error " + std::to_string(status));
            }
        }

        template <typename T>
        T device_info(cl_device_id device, cl_device_info param)
        {
            T value{};
            check(clGetDeviceInfo(device, param, sizeof(T), &value, nullptr), "clGetDeviceInfo");
            return value;
        }

        std::string device_info_string(cl_device_id device, cl_device_info param)
        {
            std::size_t size = 0;
            check(clGetDeviceInfo(device, param, 0, nullptr, &size), "clGetDeviceInfo");
            std::string value(size, '\0');
            check(clGetDeviceInfo(device, param, size, value.data(), nullptr), "clGetDeviceInfo");
            while (!value.empty() && value.back() == '\0')
            {
                value.pop_back();
            }
            return value;
        }

        cl_device_id select_device()
        {
            cl_uint platform_count = 0;
            if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0)
            {
                throw std::runtime_error("no OpenCL platform found");
            }
            std::vector<cl_platform_id> platforms(platform_count);
            check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");

            for (const cl_device_type type : {cl_device_type{CL_DEVICE_TYPE_GPU}, cl_device_type{CL_DEVICE_TYPE_ALL}})
            {
                for (const cl_platform_id platform : platforms)
                {
                    cl_uint device_count = 0;
                    if (clGetDeviceIDs(platform, type, 0, nullptr, &device_count) != CL_SUCCESS || device_count == 0)
                    {
                        continue;
                    }
                    std::vector<cl_device_id> devices(device_count);
                    check(clGetDeviceIDs(platform, type, device_count, devices.data(), nullptr), "clGetDeviceIDs");
                    return devices.front();
                }
            }
            throw std::runtime_error("no OpenCL device found");
        }

        KernelHandle create_kernel(cl_program program, const char* name)
        {
            cl_int       status = CL_SUCCESS;
            KernelHandle kernel(clCreateKernel(program, name, &status));
            check(status, "clCreateKernel");
            return kernel;
        }

        MemHandle create_buffer(cl_context context, cl_mem_flags flags, std::size_t bytes, void* host = nullptr)
        {
            cl_int    status = CL_SUCCESS;
            MemHandle buffer(clCreateBuffer(context, flags, bytes, host, &status));
            check(status, "clCreateBuffer");
            return buffer;
        }

        template <typename... Args>
        void set_args(cl_kernel kernel, const Args&... args)
        {
            cl_uint index = 0;
            (check(clSetKernelArg(kernel, index++, sizeof(Args), &args), "clSetKernelArg"), ...);
        }

        void enqueue(cl_command_queue queue, cl_kernel kernel, std::size_t work_items)
        {
            constexpr std::size_t group  = 256;
            const std::size_t     global = (work_items + group - 1) / group * group;
            check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
                  "clEnqueueNDRangeKernel");
        }

        cl_float4 to_float4(const glm::dvec3& v)
        {
            cl_float4 result{};
            result.s[0] = static_cast<cl_float>(v.x);
            result.s[1] = static_cast<cl_float>(v.y);
            result.s[2] = static_cast<cl_float>(v.z);
            return result;
        }

        // Keeps infinite bounds infinite and clamps finite ones into float range, where the cast would be undefined.
        cl_float to_clip_float(double value)
        {
            constexpr double limit = std::numeric_limits<cl_float>::max();
            return std::isinf(value) ? static_cast<cl_float>(value) : static_cast<cl_float>(std::clamp(value, -limit, limit));
        }

    }

    struct OpenCLRenderer::Impl
    {
        cl_device_id  device = nullptr;
        std::string   device_name;
        cl_ulong      max_alloc = 0;
        ContextHandle context;
        QueueHandle   queue;
        ProgramHandle program;
        KernelHandle  splat_depth;
        KernelHandle  splat_index;
        KernelHandle  resolve;
        MemHandle     points;
        cl_uint       point_count = 0;
        glm::dvec3    origin{0.0};

        MemHandle   depth;
        MemHandle   index;
        MemHandle   color;
        std::size_t pixel_capacity = 0;

        MemHandle allocate(cl_mem_flags flags, std::size_t bytes, void* host = nullptr)
        {
            if (bytes > max_alloc)
            {
                throw std::runtime_error("OpenCL buffer of " + std::to_string(bytes) + " bytes exceeds device limit of " +
                                         std::to_string(max_alloc) + " bytes");
            }
            return create_buffer(context.get(), flags, bytes, host);
        }

        void ensure_pixel_buffers(std::size_t pixel_count)
        {
            if (pixel_count <= pixel_capacity)
            {
                return;
            }
            depth.reset();
            index.reset();
            color.reset();
            pixel_capacity = 0;
            depth          = allocate(CL_MEM_READ_WRITE, pixel_count * sizeof(cl_uint));
            index          = allocate(CL_MEM_READ_WRITE, pixel_count * sizeof(cl_uint));
            color          = allocate(CL_MEM_WRITE_ONLY, pixel_count * sizeof(cl_uchar4));
            pixel_capacity = pixel_count;
        }
    };

    OpenCLRenderer::OpenCLRenderer() : impl_(std::make_unique<Impl>())
    {
        Impl& s       = *impl_;
        s.device      = select_device();
        s.device_name = device_info_string(s.device, CL_DEVICE_NAME);
        s.max_alloc   = device_info<cl_ulong>(s.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);

        const cl_platform_id        platform     = device_info<cl_platform_id>(s.device, CL_DEVICE_PLATFORM);
        const cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0};

        cl_int status = CL_SUCCESS;
        s.context.reset(clCreateContext(properties, 1, &s.device, nullptr, nullptr, &status));
        check(status, "clCreateContext");

        s.queue.reset(clCreateCommandQueue(s.context.get(), s.device, 0, &status));
        check(status, "clCreateCommandQueue");

        const char* source = kernel_source;
        s.program.reset(clCreateProgramWithSource(s.context.get(), 1, &source, nullptr, &status));
        check(status, "clCreateProgramWithSource");

        if (clBuildProgram(s.program.get(), 1, &s.device, nullptr, nullptr, nullptr) != CL_SUCCESS)
        {
            std::size_t size = 0;
            clGetProgramBuildInfo(s.program.get(), s.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
            std::string log(size, '\0');
            clGetProgramBuildInfo(s.program.get(), s.device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
            throw std::runtime_error("OpenCL program build failed:\n" + log);
        }

        s.splat_depth = create_kernel(s.program.get(), "splat_depth");
        s.splat_index = create_kernel(s.program.get(), "splat_index");
        s.resolve     = create_kernel(s.program.get(), "resolve");
    }

    OpenCLRenderer::~OpenCLRenderer() = default;

    const std::string& OpenCLRenderer::device_name() const { return impl_->device_name; }

    void OpenCLRenderer::upload(const PointCloud& cloud)
    {
        Impl& s = *impl_;
        if (cloud.points.empty())
        {
            throw std::invalid_argument("cannot upload an empty point cloud");
        }
        if (cloud.points.size() >= invalid_index)
        {
            throw std::invalid_argument("point cloud has too many points");
        }

        s.points.reset();
        s.point_count = 0;
        s.points      = s.allocate(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cloud.points.size() * sizeof(Point),
                                   const_cast<Point*>(cloud.points.data()));
        s.point_count = static_cast<cl_uint>(cloud.points.size());
        s.origin      = cloud.origin;
    }

    Image OpenCLRenderer::render(const View& view, const OrthographicCamera& camera, const RenderSettings& settings)
    {
        Impl& s = *impl_;
        if (!s.points)
        {
            throw std::logic_error("no point cloud uploaded");
        }
        if (camera.width <= 0 || camera.height <= 0 || camera.half_width <= 0.0 || camera.half_height <= 0.0)
        {
            throw std::invalid_argument("invalid orthographic camera");
        }
        if (!(view.clip_near < view.clip_far))
        {
            throw std::invalid_argument("view " + view.name + " has an empty depth range");
        }

        const std::size_t pixel_count = static_cast<std::size_t>(camera.width) * static_cast<std::size_t>(camera.height);
        if (pixel_count >= invalid_index)
        {
            throw std::invalid_argument("image is too large");
        }
        s.ensure_pixel_buffers(pixel_count);

        cl_command_queue queue = s.queue.get();
        check(clEnqueueFillBuffer(queue, s.depth.get(), &invalid_index, sizeof(invalid_index), 0,
                                  pixel_count * sizeof(cl_uint), 0, nullptr, nullptr),
              "clEnqueueFillBuffer");
        check(clEnqueueFillBuffer(queue, s.index.get(), &invalid_index, sizeof(invalid_index), 0,
                                  pixel_count * sizeof(cl_uint), 0, nullptr, nullptr),
              "clEnqueueFillBuffer");

        const cl_mem    points  = s.points.get();
        const cl_mem    depth   = s.depth.get();
        const cl_mem    index   = s.index.get();
        const cl_mem    color   = s.color.get();
        const cl_uint   count   = s.point_count;
        const cl_float4 target  = to_float4(view.target - s.origin);
        const cl_float4 right   = to_float4(view.right);
        const cl_float4 up      = to_float4(view.up);
        const cl_float4 forward = to_float4(view.forward);
        cl_float2       scale{};
        scale.s[0] = static_cast<cl_float>(camera.width / (2.0 * camera.half_width));
        scale.s[1] = static_cast<cl_float>(camera.height / (2.0 * camera.half_height));
        cl_float2 clip{};
        clip.s[0]           = to_clip_float(view.clip_near);
        clip.s[1]           = to_clip_float(view.clip_far);
        const cl_int width  = camera.width;
        const cl_int height = camera.height;
        const cl_int radius = settings.point_radius;

        set_args(s.splat_depth.get(), points, count, target, right, up, forward, scale, clip, width, height, radius, depth);
        enqueue(queue, s.splat_depth.get(), count);

        set_args(s.splat_index.get(), points, count, target, right, up, forward, scale, clip, width, height, radius, depth, index);
        enqueue(queue, s.splat_index.get(), count);

        const float    range           = settings.intensity.max - settings.intensity.min;
        const cl_float intensity_min   = settings.intensity.min;
        const cl_float intensity_scale = range > 0.0f ? 1.0f / range : 0.0f;
        const cl_uint  pixels          = static_cast<cl_uint>(pixel_count);
        const cl_int   channels        = channel_count(settings.format);
        const cl_int   colormap        = static_cast<cl_int>(settings.colormap);
        set_args(s.resolve.get(), index, points, pixels, intensity_min, intensity_scale, channels, colormap, color);
        enqueue(queue, s.resolve.get(), pixel_count);

        Image image;
        image.width    = camera.width;
        image.height   = camera.height;
        image.channels = channels;
        image.pixels.resize(pixel_count * static_cast<std::size_t>(channels));
        check(clEnqueueReadBuffer(queue, color, CL_TRUE, 0, image.pixels.size(), image.pixels.data(), 0, nullptr, nullptr),
              "clEnqueueReadBuffer");
        return image;
    }

}

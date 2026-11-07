#include <cvc/image.hpp>

#include <zlib-ng.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{

    // stb_image_write's own deflate runs at ~60 MiB/s; PNG compression goes through zlib-ng instead.
    // The pixels are unchanged, PNG is lossless.
    constexpr int png_filter       = 0;
    constexpr int deflate_level    = 1;
    constexpr int deflate_strategy = Z_RLE;

    // Matches the STBIW_ZLIB_COMPRESS signature: returns a zlib stream allocated with malloc, freed by stb.
    unsigned char* zlib_ng_compress(unsigned char* data, int length, int* out_length, int /*quality*/)
    {
        zng_stream stream{};
        if (length < 0 || zng_deflateInit2(&stream, deflate_level, Z_DEFLATED, 15, 8, deflate_strategy) != Z_OK)
        {
            return nullptr;
        }

        const std::size_t bound    = zng_deflateBound(&stream, static_cast<unsigned long>(length));
        std::size_t       capacity = std::min(bound, static_cast<std::size_t>(length) / 16 + 65536);
        auto*             out      = static_cast<unsigned char*>(std::malloc(capacity));

        stream.next_in   = data;
        stream.avail_in  = static_cast<std::uint32_t>(length);
        stream.next_out  = out;
        stream.avail_out = static_cast<std::uint32_t>(capacity);

        int result = out ? zng_deflate(&stream, Z_FINISH) : Z_MEM_ERROR;
        while ((result == Z_OK || result == Z_BUF_ERROR) && capacity < bound)
        {
            capacity    = std::min(bound, capacity * 2);
            auto* grown = static_cast<unsigned char*>(std::realloc(out, capacity));
            if (!grown)
            {
                result = Z_MEM_ERROR;
                break;
            }
            out              = grown;
            stream.next_out  = out + stream.total_out;
            stream.avail_out = static_cast<std::uint32_t>(capacity - stream.total_out);
            result           = zng_deflate(&stream, Z_FINISH);
        }

        zng_deflateEnd(&stream);
        if (result != Z_STREAM_END)
        {
            std::free(out);
            return nullptr;
        }
        *out_length = static_cast<int>(stream.total_out);
        return out;
    }

}

#define STBIW_ZLIB_COMPRESS zlib_ng_compress
#define STBIW_CRC32(buffer, length) zng_crc32(0, buffer, static_cast<std::uint32_t>(length))
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace cvc
{

    bool png_writable(int width, int height, int channels)
    {
        if (width <= 0 || height <= 0 || channels <= 0)
        {
            return false;
        }
        const std::uint64_t limit     = static_cast<std::uint64_t>(INT_MAX);
        const std::uint64_t row_bytes = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(channels) + 1;
        const std::uint64_t data      = row_bytes * static_cast<std::uint64_t>(height);
        const std::uint64_t stored    = data + 2 + ((data + 32766) / 32767) * 5;
        return data + 32766 <= limit && stored + 64 <= limit;
    }

    bool png_readable(int width, int height, int channels)
    {
        if (width <= 0 || height <= 0 || channels <= 0)
        {
            return false;
        }
        // The same test stb_image applies to the PNG header.
        return (1 << 30) / width / channels >= height;
    }

    void save_png(const Image& image, const std::filesystem::path& path)
    {
        const std::size_t expected = static_cast<std::size_t>(image.width) * image.height * image.channels;
        if (image.width <= 0 || image.height <= 0 || image.channels <= 0 || image.pixels.size() != expected)
        {
            throw std::invalid_argument("invalid image for " + path.string());
        }
        if (!png_writable(image.width, image.height, image.channels))
        {
            throw std::invalid_argument(std::to_string(image.width) + " x " + std::to_string(image.height) +
                                        " image exceeds the 2 GiB buffer limit of the PNG writer: " + path.string());
        }

        if (path.has_parent_path())
        {
            std::filesystem::create_directories(path.parent_path());
        }

        // stb tries all five row filters per row unless one is forced; the setting is a global, set once.
        static const bool filter_forced = []
        {
            stbi_write_force_png_filter = png_filter;
            return true;
        }();
        (void)filter_forced;

        const int stride = image.width * image.channels;
        if (!stbi_write_png(path.string().c_str(), image.width, image.height, image.channels, image.pixels.data(), stride))
        {
            throw std::runtime_error("failed to write " + path.string());
        }
    }

    Image load_png(const std::filesystem::path& path, int channels)
    {
        int width           = 0;
        int height          = 0;
        int source_channels = 0;

        const std::unique_ptr<stbi_uc, void (*)(void*)> pixels(
            stbi_load(path.string().c_str(), &width, &height, &source_channels, channels), stbi_image_free);
        if (!pixels)
        {
            throw std::runtime_error("failed to read " + path.string() + ": " + stbi_failure_reason());
        }

        Image image;
        image.width    = width;
        image.height   = height;
        image.channels = channels;
        image.pixels.assign(pixels.get(), pixels.get() + static_cast<std::size_t>(width) * height * channels);
        return image;
    }

}

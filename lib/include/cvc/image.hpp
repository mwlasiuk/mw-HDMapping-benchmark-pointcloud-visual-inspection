#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace cvc
{

    struct Image
    {
        int                       width    = 0;
        int                       height   = 0;
        int                       channels = 0;
        std::vector<std::uint8_t> pixels;
    };

    bool png_writable(int width, int height, int channels);

    // Whether load_png can read such a PNG back; the reader refuses images over 1 GiB of raw pixels.
    bool png_readable(int width, int height, int channels);

    void save_png(const Image& image, const std::filesystem::path& path);

    Image load_png(const std::filesystem::path& path, int channels);

}

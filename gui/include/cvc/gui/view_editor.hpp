#pragma once

#include <cvc/bounds.hpp>
#include <cvc/camera.hpp>
#include <cvc/point_cloud.hpp>

#include <functional>
#include <string>
#include <vector>

namespace cvc::gui
{

    // What the run would make of a shot, shown next to it in the editor.
    struct ShotStatus
    {
        std::string summary; // Output sizes, one item per line.
        std::string warning; // The shot renders, but a later step will fail or degrade.
        std::string error;   // The shot cannot be rendered; the editor refuses to finish until it is fixed.
    };

    struct EditorScene
    {
        const PointCloud* reference = nullptr; // Drawn as every 10th point.
        Bounds            bounds;              // Of the reference.
        int               resolution = 0;      // Pixels along the longer side of every render.

        // Fits the image rectangle of a view to the reference within its depth range; throws when nothing is inside.
        std::function<Shot(const View&)>       fit;
        std::function<ShotStatus(const Shot&)> check;
    };

    // Opens a window to edit the position, orientation, extent and depth range of the shots, and blocks until it
    // closes. Closing the window keeps the edits; returns false only when the user aborts the run instead. Edited
    // shots get cameras of scene.resolution pixels along their longer side; shots left alone are returned unchanged.
    // Throws when no window can be opened.
    bool edit_views(const EditorScene& scene, std::vector<Shot>& shots);

}

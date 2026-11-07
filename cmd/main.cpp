#include <cvc/bounds.hpp>
#include <cvc/camera.hpp>
#include <cvc/color.hpp>
#include <cvc/image.hpp>
#include <cvc/json.hpp>
#include <cvc/mosaic.hpp>
#include <cvc/opencl_renderer.hpp>
#include <cvc/point_cloud.hpp>
#include <cvc/run_log.hpp>

#ifdef CVC_WITH_GUI
    #include <cvc/gui/view_editor.hpp>
#endif

#include <CLI/CLI.hpp>
#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

    class Stopwatch
    {
    public:
        double seconds() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count(); }

    private:
        std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    };

    class PngWriter
    {
    public:
        using SavedCallback = std::function<void(const std::filesystem::path&)>;

        // on_saved runs on the encoding thread after each file is written.
        PngWriter(std::size_t workers, std::size_t budget, SavedCallback on_saved)
            : workers_(workers), budget_(budget), on_saved_(std::move(on_saved)) {}

        void write(cvc::Image image, std::filesystem::path path)
        {
            const std::size_t bytes = image.pixels.size();
            while (!pending_.empty() && (pending_.size() >= workers_ || pending_bytes_ + bytes > budget_))
            {
                finish_oldest();
            }
            pending_bytes_ += bytes;
            pending_.push_back({std::async(std::launch::async, [image = std::move(image), path = std::move(path), on_saved = on_saved_]
                                           {
                                               cvc::save_png(image, path);
                                               on_saved(path); }),
                                bytes});
        }

        void wait()
        {
            while (!pending_.empty())
            {
                finish_oldest();
            }
        }

    private:
        struct Job
        {
            std::future<void> done;
            std::size_t       bytes = 0;
        };

        void finish_oldest()
        {
            Job job = std::move(pending_.front());
            pending_.pop_front();
            pending_bytes_ -= job.bytes;
            job.done.get();
        }

        std::size_t     workers_;
        std::size_t     budget_;
        SavedCallback   on_saved_;
        std::size_t     pending_bytes_ = 0;
        std::deque<Job> pending_;
    };

    struct Options
    {
        std::filesystem::path              reference;
        std::vector<std::filesystem::path> compare;
        std::filesystem::path              output_dir;
        int                                resolution        = 16384;
        int                                mosaic_resolution = 0;
        int                                point_radius      = 0;
        double                             fit_fraction      = 0.9999;
        cvc::PixelFormat                   format            = cvc::PixelFormat::grayscale_alpha;
        cvc::Colormap                      colormap          = cvc::Colormap::viridis;
        bool                               resume            = false;
        bool                               gui_config        = false;
    };

    using Json = nlohmann::ordered_json;

    // Bumped when the settings file changes meaning, so older files are not resumed.
    constexpr int settings_version = 1;

    const char* const settings_file_name = "settings.json";

    const std::map<std::string, cvc::PixelFormat>& format_names()
    {
        static const std::map<std::string, cvc::PixelFormat> names{
            {"grayscale", cvc::PixelFormat::grayscale},
            {"grayscale_alpha", cvc::PixelFormat::grayscale_alpha},
            {"color", cvc::PixelFormat::color},
            {"color_alpha", cvc::PixelFormat::color_alpha},
        };
        return names;
    }

    const std::map<std::string, cvc::Colormap>& colormap_names()
    {
        static const std::map<std::string, cvc::Colormap> names{
            {"viridis", cvc::Colormap::viridis},
            {"turbo", cvc::Colormap::turbo},
            {"inferno", cvc::Colormap::inferno},
            {"plasma", cvc::Colormap::plasma},
            {"cividis", cvc::Colormap::cividis},
            {"magma", cvc::Colormap::magma},
            {"cubehelix", cvc::Colormap::cubehelix},
            {"hot", cvc::Colormap::hot},
            {"blues", cvc::Colormap::blues},
            {"blue_white_red", cvc::Colormap::blue_white_red},
        };
        return names;
    }

    template <typename T>
    std::string name_of(const std::map<std::string, T>& names, T value)
    {
        for (const auto& [name, candidate] : names)
        {
            if (candidate == value)
            {
                return name;
            }
        }
        return "unknown";
    }

    bool uses_colormap(cvc::PixelFormat format) { return cvc::channel_count(format) >= 3; }

    struct ViewPlan
    {
        cvc::Shot       shot;
        cvc::MosaicPlan mosaic;
    };

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ViewPlan, shot, mosaic)

    struct LoadedCloud
    {
        cvc::PointCloud cloud;
        cvc::Bounds     bounds;
    };

    struct RenderJob
    {
        const cvc::View*               view;
        const cvc::OrthographicCamera* camera;
        std::filesystem::path          path;
    };

    // Views and written outputs of an earlier run with the same settings.
    struct Resume
    {
        std::vector<ViewPlan>    plans;
        std::vector<std::string> completed;
    };

    std::string format(const glm::dvec3& v)
    {
        std::ostringstream out;
        out << std::fixed << std::setprecision(3) << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return out.str();
    }

    std::string size_text(int width, int height) { return std::to_string(width) + " x " + std::to_string(height) + " px"; }

    std::string describe(const cvc::OrthographicCamera& camera)
    {
        std::ostringstream out;
        out << size_text(camera.width, camera.height) << ", " << std::fixed << std::setprecision(3)
            << 2.0 * camera.half_width << " x " << 2.0 * camera.half_height << " m, " << std::setprecision(5)
            << 2.0 * camera.half_width / camera.width << " m/px";
        return out.str();
    }

    std::string directory_label(const std::filesystem::path& path)
    {
        const std::filesystem::path directory = std::filesystem::absolute(path).parent_path().filename();
        return directory.empty() ? path.stem().string() : directory.string();
    }

    std::vector<std::string> make_labels(const std::vector<std::filesystem::path>& paths)
    {
        std::vector<std::string> directories;
        for (const std::filesystem::path& path : paths)
        {
            directories.push_back(directory_label(path));
        }

        std::vector<std::string> labels = directories;
        for (std::size_t i = 0; i < labels.size(); ++i)
        {
            if (std::count(directories.begin(), directories.end(), directories[i]) > 1)
            {
                labels[i] += "_" + paths[i].stem().string();
            }
        }

        for (const std::string& label : labels)
        {
            if (std::count(labels.begin(), labels.end(), label) > 1)
            {
                throw std::runtime_error("output name '" + label + "' is shared by more than one point cloud");
            }
        }
        return labels;
    }

    std::filesystem::path png_path(const std::filesystem::path& directory, const std::string& name)
    {
        return (directory / (name + ".png")).make_preferred();
    }

    // Margin around the reference points a fitted view frames.
    constexpr double fit_margin = 1.02;

    // Lays out the mosaic of a shot; throws when its render or mosaic exceeds the PNG writer limit.
    ViewPlan plan_view(const cvc::Shot& shot, std::size_t count, const Options& options)
    {
        const ViewPlan plan{shot, options.mosaic_resolution > 0
                                      ? cvc::plan_mosaic(count, shot.camera, options.mosaic_resolution, options.format)
                                      : cvc::plan_largest_mosaic(count, shot.camera, options.format)};
        if (!cvc::png_writable(shot.camera.width, shot.camera.height, cvc::channel_count(options.format)))
        {
            throw std::runtime_error("render of view " + shot.view.name + " (" +
                                     size_text(shot.camera.width, shot.camera.height) +
                                     ") exceeds the 2 GiB PNG writer limit, lower --resolution");
        }
        const cvc::MosaicLayout& layout = plan.mosaic.layout;
        if (!cvc::png_writable(plan.mosaic))
        {
            throw std::runtime_error("mosaic of view " + shot.view.name + " (" + size_text(layout.width(), layout.height()) +
                                     ", tiles " + size_text(layout.tile_width, layout.tile_height) +
                                     ") exceeds the 2 GiB PNG writer limit, lower --mosaic-resolution");
        }
        return plan;
    }

    // The mosaic is built from tiles read back from disk, which fails for tiles the PNG reader refuses.
    std::string mosaic_warning(const ViewPlan& plan)
    {
        const cvc::MosaicLayout& layout = plan.mosaic.layout;
        if (cvc::png_readable(layout.tile_width, layout.tile_height, plan.mosaic.tile_channels))
        {
            return {};
        }
        return "mosaic tiles of view " + plan.shot.view.name + " (" + size_text(layout.tile_width, layout.tile_height) +
               ") exceed the 1 GiB PNG reader limit, so building its mosaic will fail after rendering; lower "
               "--mosaic-resolution";
    }

    std::vector<ViewPlan> plan_views(const cvc::PointCloud& reference, const cvc::Bounds& bounds, std::size_t count,
                                     const Options& options)
    {
        std::vector<ViewPlan> plans;
        for (const cvc::Shot& shot :
             cvc::fit_views(reference, cvc::make_views(bounds), options.resolution, fit_margin, options.fit_fraction))
        {
            plans.push_back(plan_view(shot, count, options));
        }
        return plans;
    }

    // Depth limits of a view along its direction from its center, or nothing when it has none.
    std::string depth_text(const cvc::View& view)
    {
        if (std::isinf(view.clip_near) && std::isinf(view.clip_far))
        {
            return {};
        }
        const auto limit = [](double value)
        {
            if (std::isinf(value))
            {
                return std::string("unlimited");
            }
            std::ostringstream out;
            out << std::fixed << std::setprecision(3) << value << " m";
            return out.str();
        };
        return "depth " + limit(view.clip_near) + " to " + limit(view.clip_far);
    }

    void print_views(const std::vector<ViewPlan>& plans)
    {
        for (const ViewPlan& plan : plans)
        {
            const cvc::MosaicLayout& layout = plan.mosaic.layout;
            std::cout << "  " << std::left << std::setw(9) << plan.shot.view.name << std::right << " "
                      << describe(plan.shot.camera) << ", mosaic " << size_text(layout.width(), layout.height()) << " ("
                      << layout.columns << " x " << layout.rows << " tiles of "
                      << size_text(layout.tile_width, layout.tile_height) << ")";
            const std::string depth = depth_text(plan.shot.view);
            if (!depth.empty())
            {
                std::cout << ", " << depth;
            }
            std::cout << "\n";
            const std::string warning = mosaic_warning(plan);
            if (!warning.empty())
            {
                std::cout << "  warning: " << warning << "\n";
            }
        }
    }

    std::string utf8(const std::filesystem::path& path) { return path.u8string(); }

    // Everything the outputs depend on besides the code: the input files and the options. A run resumes only when
    // these match; the views are derived from them and stored next to them.
    Json describe_settings(const std::vector<std::filesystem::path>& paths, const std::vector<std::string>& labels,
                           const Options& options)
    {
        Json inputs = Json::array();
        for (std::size_t i = 0; i < paths.size(); ++i)
        {
            const std::filesystem::path path = std::filesystem::weakly_canonical(paths[i]);
            inputs.push_back(Json{{"label", utf8(labels[i])},
                                  {"path", utf8(path)},
                                  {"size", std::filesystem::file_size(path)},
                                  {"write_time", std::filesystem::last_write_time(path).time_since_epoch().count()}});
        }

        Json recorded{{"resolution", options.resolution},
                      {"mosaic_resolution", options.mosaic_resolution},
                      {"point_radius", options.point_radius},
                      {"fit_fraction", options.fit_fraction},
                      {"format", name_of(format_names(), options.format)}};
        if (uses_colormap(options.format))
        {
            recorded["colormap"] = name_of(colormap_names(), options.colormap);
        }

        return Json{{"version", settings_version}, {"inputs", std::move(inputs)}, {"options", std::move(recorded)}};
    }

    void print_changes(const Json& before, const Json& after)
    {
        for (const Json& change : Json::diff(before, after))
        {
            const std::string operation = change.at("op").get<std::string>();
            const std::string pointer   = change.at("path").get<std::string>();
            std::cout << "  " << pointer << ": ";
            if (operation == "replace")
            {
                std::cout << before.at(Json::json_pointer(pointer)).dump() << " -> " << change.at("value").dump();
            }
            else if (operation == "add")
            {
                std::cout << "added " << change.at("value").dump();
            }
            else
            {
                std::cout << "removed";
            }
            std::cout << "\n";
        }
    }

    std::optional<Resume> read_resume(const std::filesystem::path& file, const Json& settings)
    {
        std::ifstream in(file);
        if (!in)
        {
            std::cout << "Nothing to resume, " << file.string() << " not found, starting from the beginning\n";
            return std::nullopt;
        }

        try
        {
            const Json stored          = Json::parse(in);
            Json       stored_settings = stored;
            stored_settings.erase("views");
            stored_settings.erase("completed");
            if (stored_settings != settings)
            {
                std::cout << "Settings differ from " << file.string() << ", starting from the beginning:\n";
                print_changes(stored_settings, settings);
                return std::nullopt;
            }
            return Resume{stored.at("views").get<std::vector<ViewPlan>>(), stored.at("completed").get<std::vector<std::string>>()};
        }
        catch (const Json::exception& e)
        {
            std::cout << "Cannot resume from " << file.string() << " (" << e.what() << "), starting from the beginning\n";
            return std::nullopt;
        }
    }

    // Opens the view editor on the plans. Returns the indices of the views the user changed, which get new plans
    // while the others keep theirs exactly, or nothing when the run is aborted in the editor.
    std::optional<std::vector<std::size_t>> edit_plans(const LoadedCloud& reference, std::size_t count,
                                                       const Options& options, std::vector<ViewPlan>& plans)
    {
#ifdef CVC_WITH_GUI
        std::vector<cvc::Shot> shots;
        for (const ViewPlan& plan : plans)
        {
            shots.push_back(plan.shot);
        }

        cvc::gui::EditorScene scene;
        scene.reference  = &reference.cloud;
        scene.bounds     = reference.bounds;
        scene.resolution = options.resolution;
        scene.fit        = [&](const cvc::View& view)
        { return cvc::fit_view(reference.cloud, view, options.resolution, fit_margin, options.fit_fraction); };
        scene.check = [&](const cvc::Shot& shot)
        {
            cvc::gui::ShotStatus status;
            try
            {
                const ViewPlan           plan   = plan_view(shot, count, options);
                const cvc::MosaicLayout& layout = plan.mosaic.layout;
                status.summary                  = "render " + describe(shot.camera) + "\nmosaic " + size_text(layout.width(), layout.height()) +
                                 ", " + std::to_string(layout.columns) + " x " + std::to_string(layout.rows) +
                                 " tiles of " + size_text(layout.tile_width, layout.tile_height);
                status.warning = mosaic_warning(plan);
            }
            catch (const std::exception& e)
            {
                status.error = e.what();
            }
            return status;
        };

        std::cout << "View editor open, close its window to render with the views it shows\n";
        if (!cvc::gui::edit_views(scene, shots))
        {
            return std::nullopt;
        }

        std::vector<std::size_t> changed;
        for (std::size_t i = 0; i < plans.size(); ++i)
        {
            if (Json(shots[i]) != Json(plans[i].shot))
            {
                plans[i] = plan_view(shots[i], count, options);
                changed.push_back(i);
            }
        }
        return changed;
#else
        (void)reference;
        (void)count;
        (void)options;
        (void)plans;
        throw std::logic_error("this build has no view editor");
#endif
    }

    // Drops the outputs of the given views from the outputs recorded as written, so they are rendered again.
    std::size_t forget_outputs(std::vector<std::string>& completed, const std::filesystem::path& output_dir,
                               const std::vector<std::filesystem::path>& outputs)
    {
        const auto normal = [](const std::filesystem::path& path)
        { return std::filesystem::absolute(path).lexically_normal(); };

        std::set<std::filesystem::path> stale;
        for (const std::filesystem::path& output : outputs)
        {
            stale.insert(normal(output));
        }
        const std::size_t before = completed.size();
        completed.erase(std::remove_if(completed.begin(), completed.end(), [&](const std::string& entry)
                                       { return stale.count(normal(output_dir / std::filesystem::u8path(entry))) > 0; }),
                        completed.end());
        return before - completed.size();
    }

    // Returns false when the run is aborted in the view editor.
    bool run(const Options& options)
    {
        const Stopwatch total;

        std::vector<std::filesystem::path> paths{options.reference};
        paths.insert(paths.end(), options.compare.begin(), options.compare.end());
        const std::vector<std::string> labels        = make_labels(paths);
        const std::filesystem::path    full_dir      = options.output_dir / "full";
        const std::filesystem::path    tile_dir      = options.output_dir / "tile";
        const std::filesystem::path    mosaic_dir    = options.output_dir / "mosaic";
        const std::filesystem::path    settings_file = (options.output_dir / settings_file_name).make_preferred();

        cvc::OpenCLRenderer renderer;
        std::cout << "OpenCL device: " << renderer.device_name() << "\n";
        std::cout << "Format: " << name_of(format_names(), options.format);
        if (uses_colormap(options.format))
        {
            std::cout << ", colormap " << name_of(colormap_names(), options.colormap);
        }
        std::cout << "\n";

        const Json            run_settings = describe_settings(paths, labels, options);
        std::optional<Resume> resume       = options.resume ? read_resume(settings_file, run_settings) : std::nullopt;

        const auto load = [&](std::size_t i)
        {
            std::cout << "[" << i + 1 << "/" << paths.size() << "] " << labels[i] << ": "
                      << std::filesystem::path(paths[i]).make_preferred().string() << "\n";
            const Stopwatch load_watch;
            LoadedCloud     loaded{cvc::load_ply(paths[i]), {}};
            loaded.bounds = cvc::compute_bounds(loaded.cloud);
            std::cout << "  points: " << loaded.cloud.points.size() << " (" << load_watch.seconds() << " s)\n";
            std::cout << "  bounds min: " << format(loaded.bounds.min) << "\n";
            std::cout << "  bounds max: " << format(loaded.bounds.max) << "\n";
            return loaded;
        };

        // A fresh run fits the views to the reference, which is then rendered first; a resumed run reuses the views.
        std::optional<LoadedCloud> reference;
        std::vector<ViewPlan>      plans;
        if (resume)
        {
            plans = std::move(resume->plans);
            std::cout << "Resuming the run recorded in " << settings_file.string() << ", views:\n";
        }
        else
        {
            reference = load(0);
            const Stopwatch fit_watch;
            plans = plan_views(reference->cloud, reference->bounds, paths.size(), options);
            std::cout << "Views fitted to " << 100.0 * options.fit_fraction << "% of " << labels[0] << " ("
                      << fit_watch.seconds() << " s):\n";
        }
        print_views(plans);

        // Edited views are recorded in the settings file like fitted ones, so a resumed run renders them again
        // without the editor.
        if (options.gui_config)
        {
            if (!reference)
            {
                reference = load(0);
            }
            const std::optional<std::vector<std::size_t>> changed = edit_plans(*reference, paths.size(), options, plans);
            if (!changed)
            {
                std::cout << "Run aborted in the view editor, nothing rendered\n";
                return false;
            }
            if (changed->empty())
            {
                std::cout << "No views edited\n";
            }
            else
            {
                std::vector<ViewPlan>              edited;
                std::vector<std::filesystem::path> outputs;
                for (const std::size_t k : *changed)
                {
                    const ViewPlan& plan = plans[k];
                    edited.push_back(plan);
                    for (const std::string& label : labels)
                    {
                        outputs.push_back(png_path(full_dir, label + "_" + plan.shot.view.name));
                        outputs.push_back(png_path(tile_dir, label + "_" + plan.shot.view.name));
                    }
                    outputs.push_back(png_path(mosaic_dir, plan.shot.view.name));
                }
                std::cout << "Edited views:\n";
                print_views(edited);
                if (resume)
                {
                    const std::size_t forgotten = forget_outputs(resume->completed, options.output_dir, outputs);
                    std::cout << forgotten << " written outputs of the edited views will be rendered again\n";
                }
            }
        }

        // Written before any output, so the outputs left by a run with other settings are never taken as done.
        Json document     = run_settings;
        document["views"] = plans;
        cvc::RunLog log(settings_file, std::move(document), resume ? resume->completed : std::vector<std::string>{});
        if (resume)
        {
            std::cout << log.done_count() << " of " << plans.size() * (2 * paths.size() + 1)
                      << " outputs already written, rendering the rest\n";
        }

        PngWriter writer(std::clamp<std::size_t>(std::thread::hardware_concurrency(), 1, 4), std::size_t{3} << 30,
                         [&log](const std::filesystem::path& path)
                         {
                             try
                             {
                                 log.mark_done(path);
                             }
                             catch (const std::exception& e)
                             {
                                 std::cerr << "warning: " + std::string(e.what()) + "\n";
                             }
                         });

        std::size_t rendered = 0;
        for (std::size_t i = 0; i < paths.size(); ++i)
        {
            std::vector<RenderJob> jobs;
            for (const ViewPlan& plan : plans)
            {
                const std::string name = labels[i] + "_" + plan.shot.view.name;
                for (RenderJob job : {RenderJob{&plan.shot.view, &plan.shot.camera, png_path(full_dir, name)},
                                      RenderJob{&plan.shot.view, &plan.mosaic.tile_camera, png_path(tile_dir, name)}})
                {
                    if (!log.done(job.path))
                    {
                        jobs.push_back(std::move(job));
                    }
                }
            }
            const std::size_t written = 2 * plans.size() - jobs.size();
            if (jobs.empty())
            {
                std::cout << "[" << i + 1 << "/" << paths.size() << "] " << labels[i] << ": all " << written
                          << " renders and mosaic tiles already written, skipped\n";
                // The reference may have been loaded for the view editor; release it before the next cloud loads.
                reference.reset();
                continue;
            }

            cvc::RenderSettings settings;
            settings.point_radius = options.point_radius;
            settings.format       = options.format;
            settings.colormap     = options.colormap;
            {
                const LoadedCloud loaded = i == 0 && reference ? std::move(*reference) : load(i);
                reference.reset();

                settings.intensity = cvc::intensity_range(loaded.cloud, 0.01, 0.99);
                std::cout << "  intensity range: [" << settings.intensity.min << ", " << settings.intensity.max << "]\n";

                renderer.upload(loaded.cloud);
            }

            const Stopwatch render_watch;
            for (RenderJob& job : jobs)
            {
                writer.write(renderer.render(*job.view, *job.camera, settings), std::move(job.path));
            }
            rendered += jobs.size();
            std::cout << "  rendered ";
            if (written == 0)
            {
                std::cout << plans.size() << " views and mosaic tiles";
            }
            else
            {
                std::cout << jobs.size() << " missing renders and mosaic tiles, " << written << " already written";
            }
            std::cout << " (" << render_watch.seconds() << " s)\n";
        }

        const Stopwatch render_save_watch;
        writer.wait();
        std::cout << "Saved " << rendered << " renders and tiles to " << full_dir.string() << " and " << tile_dir.string()
                  << " (waited " << render_save_watch.seconds() << " s for encoding)\n";

        std::size_t mosaics = 0;
        std::cout << "Building mosaics from " << tile_dir.string() << "\n";
        for (const ViewPlan& plan : plans)
        {
            std::filesystem::path path = png_path(mosaic_dir, plan.shot.view.name);
            if (log.done(path))
            {
                std::cout << "  " << std::left << std::setw(9) << plan.shot.view.name << std::right << " -> "
                          << path.string() << " (already written)\n";
                continue;
            }

            const Stopwatch mosaic_watch;
            cvc::Mosaic     mosaic(plan.mosaic, paths.size());
            for (std::size_t i = 0; i < paths.size(); ++i)
            {
                mosaic.place(i,
                             cvc::load_png(png_path(tile_dir, labels[i] + "_" + plan.shot.view.name),
                                           plan.mosaic.tile_channels),
                             labels[i]);
            }

            std::cout << "  " << std::left << std::setw(9) << plan.shot.view.name << std::right << " -> " << path.string()
                      << " (" << mosaic_watch.seconds() << " s)\n";
            writer.write(mosaic.release(), std::move(path));
            ++mosaics;
        }

        const Stopwatch mosaic_save_watch;
        writer.wait();
        std::cout << "Saved " << mosaics << " mosaics to " << mosaic_dir.string() << " (waited "
                  << mosaic_save_watch.seconds() << " s for encoding)\n";
        std::cout << "Done in " << total.seconds() << " s\n";
        return true;
    }

}

int main(int argc, char** argv)
{
    Options options;

    CLI::App app{"Renders a reference point cloud and point clouds registered to it from the same 14 orthographic views into PNG images and per-view mosaics"};
    app.add_option("--reference", options.reference, "Reference point cloud (PLY with x, y, z, intensity)")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--compare", options.compare, "Point clouds registered to the reference, rendered with the reference views")
        ->check(CLI::ExistingFile);
    app.add_option("--output-dir", options.output_dir, "Output directory; renders go to full, mosaic tiles to tile and mosaics to mosaic inside it, and settings.json records the run for --resume")
        ->required();
    app.add_option("--resolution", options.resolution, "Render size in pixels along the longer axis, limited by the 2 GiB PNG writer buffer")
        ->check(CLI::Range(1024, 65535))
        ->capture_default_str();
    app.add_option("--mosaic-resolution", options.mosaic_resolution, "Mosaic size in pixels along the longer axis, 0 picks the largest size the 2 GiB PNG writer buffer allows")
        ->check(CLI::Range(0, 65535))
        ->capture_default_str();
    app.add_option("--point-radius", options.point_radius, "Point splat radius in pixels")
        ->check(CLI::Range(0, 8))
        ->capture_default_str();
    app.add_option("--fit-fraction", options.fit_fraction, "Minimum fraction of reference points inside every view (1 keeps all)")
        ->check(CLI::Range(0.5, 1.0))
        ->default_str("0.9999");
    app.add_option("--format", options.format, "Pixel format of renders and mosaics: grayscale, grayscale_alpha, color or color_alpha")
        ->transform(CLI::CheckedTransformer(format_names(), CLI::ignore_case))
        ->default_str("grayscale_alpha");
    CLI::Option* colormap = app.add_option("--colormap", options.colormap, "Intensity colormap for the color and color_alpha formats")
                                ->transform(CLI::CheckedTransformer(colormap_names(), CLI::ignore_case))
                                ->default_str("viridis");
    app.add_flag("--resume", options.resume, "Continue an interrupted run: when settings.json in the output directory records the same inputs and options, keep the outputs it lists as written and render only the missing ones");
    app.add_flag("--gui-config", options.gui_config, "Open a window to adjust the views after they are fitted to the reference: position, orientation, width, height and near and far depth limits, e.g. to render a small section of a large cloud in detail. Closing the window renders with the views it shows, each --resolution pixels along its longer side. Edited views are recorded in settings.json, so --resume continues without the window; with --resume the window starts from the recorded views and the outputs of views changed in it are rendered again");
    CLI11_PARSE(app, argc, argv);

#ifndef CVC_WITH_GUI
    if (options.gui_config)
    {
        return app.exit(CLI::ValidationError("--gui-config", "this build has no view editor, configure CMake with -DCVC_BUILD_GUI=ON"));
    }
#endif

    if (colormap->count() > 0 && !uses_colormap(options.format))
    {
        return app.exit(CLI::ValidationError("--colormap", "applies only to --format color or color_alpha"));
    }

    try
    {
        // A run aborted in the view editor exits with 2, so scripts do not take it for a finished one.
        return run(options) ? 0 : 2;
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

#include <cvc/run_log.hpp>

#include <fstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace cvc
{

    RunLog::RunLog(const std::filesystem::path& file, nlohmann::ordered_json settings, const std::vector<std::string>& completed)
        : file_(std::filesystem::absolute(file).lexically_normal()), directory_(file_.parent_path()), settings_(std::move(settings))
    {
        for (const std::string& output : completed)
        {
            std::error_code error;
            if (std::filesystem::is_regular_file(directory_ / std::filesystem::u8path(output), error))
            {
                completed_.insert(output);
            }
        }

        std::filesystem::create_directories(directory_);
        save();
    }

    bool RunLog::done(const std::filesystem::path& output) const
    {
        const std::string                 name = key(output);
        const std::lock_guard<std::mutex> lock(mutex_);
        return completed_.count(name) > 0;
    }

    std::size_t RunLog::done_count() const
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return completed_.size();
    }

    void RunLog::mark_done(const std::filesystem::path& output)
    {
        const std::string                 name = key(output);
        const std::lock_guard<std::mutex> lock(mutex_);
        completed_.insert(name);
        save();
    }

    std::string RunLog::key(const std::filesystem::path& output) const
    {
        return std::filesystem::absolute(output).lexically_normal().lexically_relative(directory_).generic_u8string();
    }

    void RunLog::save() const
    {
        nlohmann::ordered_json document = settings_;
        document["completed"]           = completed_;

        std::filesystem::path temporary = file_;
        temporary += ".tmp";
        {
            std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
            out << document.dump(2) << '\n';
            out.close();
            if (!out)
            {
                throw std::runtime_error("failed to write " + temporary.string());
            }
        }
        std::filesystem::rename(temporary, file_);
    }

}

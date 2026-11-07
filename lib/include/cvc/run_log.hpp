#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace cvc
{

    // A JSON file holding the settings of a run and the outputs it has finished writing, so an interrupted run can
    // continue where it stopped. Outputs are recorded relative to the directory of the file, under "completed".
    // The file is replaced through a temporary file and a rename, so an interruption never leaves it half written.
    class RunLog
    {
    public:
        // Keeps the completed outputs whose files still exist and writes the file; throws when it cannot be written.
        RunLog(const std::filesystem::path& file, nlohmann::ordered_json settings, const std::vector<std::string>& completed);

        RunLog(const RunLog&)            = delete;
        RunLog& operator=(const RunLog&) = delete;

        bool done(const std::filesystem::path& output) const;

        std::size_t done_count() const;

        // Thread safe. Throws when the file cannot be written; the output stays recorded and is saved by the next call.
        void mark_done(const std::filesystem::path& output);

    private:
        std::string key(const std::filesystem::path& output) const;

        void save() const;

        std::filesystem::path  file_;
        std::filesystem::path  directory_;
        nlohmann::ordered_json settings_;
        std::set<std::string>  completed_;
        mutable std::mutex     mutex_;
    };

}

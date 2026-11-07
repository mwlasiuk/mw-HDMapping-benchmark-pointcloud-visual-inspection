import os
import re
import shutil
import subprocess
import sys
import sysconfig

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CLANG_FORMAT_EXECUTABLE = "clang-format"
CLANG_FORMAT_PACKAGE = "clang-format"

REQUIREMENTS_FILE = os.path.join(PROJECT_DIR, "requirements.txt")

FILE_LOCATIONS = [
    os.path.join(PROJECT_DIR, "cmd"),
    os.path.join(PROJECT_DIR, "gui"),
    os.path.join(PROJECT_DIR, "lib"),
]

FILE_EXCLUDE_LOCATIONS = [
    os.path.join(PROJECT_DIR, "external"),
    os.path.join(PROJECT_DIR, "build"),
]

FILE_EXTENSIONS = [
    ".c",
    ".h",
    ".cpp",
    ".hpp",
]


class ClangFormatError(RuntimeError):
    pass


def _get_fix_instructions() -> str:
    requirements_path = os.path.relpath(REQUIREMENTS_FILE, PROJECT_DIR)

    if sys.prefix != sys.base_prefix:
        return f"Install the pinned version into this virtual environment with:\n  {sys.executable} -m pip install -r {requirements_path}"

    python_path = os.path.join(".venv", "Scripts" if os.name == "nt" else "bin", "python")

    return (
        "Set up the pinned clang-format once, from the repository root:\n"
        "  python -m venv .venv\n"
        f"  {python_path} -m pip install -r {requirements_path}\n"
        f"then run the clang-format scripts with {python_path}"
    )


def get_pinned_clang_format_version() -> str:
    pattern = re.compile(rf"^{re.escape(CLANG_FORMAT_PACKAGE)}\s*==\s*([0-9][0-9.]*)$")

    with open(REQUIREMENTS_FILE, encoding="utf-8") as requirements:
        for line in requirements:
            match = pattern.match(line.split("#", 1)[0].strip())

            if match:
                return match.group(1)

    requirements_path = os.path.relpath(REQUIREMENTS_FILE, PROJECT_DIR)
    raise ClangFormatError(f"{requirements_path} has no '{CLANG_FORMAT_PACKAGE}==<version>' line")


def get_clang_format_version(clang_format: str) -> str | None:
    result = subprocess.run(
        [clang_format, "--version"],
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        return None

    match = re.search(r"clang-format version (\d+\.\d+\.\d+)", result.stdout)
    return match.group(1) if match else None


def find_clang_format() -> str | None:
    return shutil.which(CLANG_FORMAT_EXECUTABLE, path=sysconfig.get_path("scripts"))


def find_pinned_clang_format() -> tuple[str, str]:
    pinned_version = get_pinned_clang_format_version()
    clang_format = find_clang_format()

    if clang_format is None:
        raise ClangFormatError(
            f"{CLANG_FORMAT_EXECUTABLE} is not installed in this Python environment ({sys.prefix})\n"
            + _get_fix_instructions()
        )

    version = get_clang_format_version(clang_format)

    if version != pinned_version:
        requirements_path = os.path.relpath(REQUIREMENTS_FILE, PROJECT_DIR)
        raise ClangFormatError(
            f"{clang_format} is version {version or 'unknown'}, but {requirements_path} pins {pinned_version}\n"
            + _get_fix_instructions()
        )

    return clang_format, version


def is_accepted_file(file_name: str) -> bool:
    return file_name.endswith(tuple(FILE_EXTENSIONS))


def is_excluded_directory(directory_path: str) -> bool:
    directory_path = os.path.abspath(directory_path)

    return any(
        os.path.commonpath([directory_path, os.path.abspath(exclude_path)]) == os.path.abspath(exclude_path)
        for exclude_path in FILE_EXCLUDE_LOCATIONS
    )


def list_files() -> tuple[list[str], list[str]]:
    accepted_files = []
    excluded_files = []

    for location in FILE_LOCATIONS:
        for dirpath, dirnames, filenames in os.walk(location):
            if is_excluded_directory(dirpath):
                for filename in filenames:
                    if is_accepted_file(filename):
                        excluded_files.append(os.path.join(dirpath, filename))

                dirnames.clear()
                continue

            for filename in filenames:
                if is_accepted_file(filename):
                    accepted_files.append(os.path.join(dirpath, filename))

    return accepted_files, excluded_files


def get_accepted_files() -> list[str]:
    return list_files()[0]

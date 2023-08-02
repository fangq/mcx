"""
The original code was taken from Pybind11's CMake Example: https://github.com/pybind/cmake_example.git
"""
import os
import re
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

with open("README.md", "r") as fh:
    readme = fh.read()

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class CMakeExtension(Extension):
    def __init__(self, name, target, source_dir=""):
        Extension.__init__(self, name, sources=[])
        self.source_dir = os.path.abspath(source_dir)
        self.target = target


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args = [
            f"-DPython3_ROOT_DIR={sys.exec_prefix}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            "-DBUILD_PYTHON=true"
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401

                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass

        else:

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(["cmake", ext.source_dir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--target", ext.target] + build_args, cwd=build_temp)


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="pmcx",
    packages=['pmcx'],
    version="0.0.13",
    requires=['numpy'],
    license='GPLv3+',
    author="Matin Raayai Ardakani, Qianqian Fang",
    author_email="raayaiardakani.m@northeastern.edu, q.fang@neu.edu",
    description="Python bindings for Monte Carlo eXtreme photon transport simulator",
    long_description=readme,
    long_description_content_type="text/markdown",
    maintainer= 'Qianqian Fang',
    url='https://github.com/fangq/mcx',
    download_url='http://mcx.space',
    keywords=['Monte Carlo simulation', 'Biophotonics', 'Ray-tracing', 'Rendering', 'GPU', 'Modeling',
                'Biomedical Optics', 'Tissue Optics', 'Simulator', 'Optics'],
    ext_modules=[CMakeExtension("_pmcx", target="_pmcx", source_dir="../src/")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.6",
    classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'Environment :: GPU :: NVIDIA CUDA',
      'Topic :: Scientific/Engineering :: Physics',
      'License :: OSI Approved :: Apache Software License',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      'Programming Language :: Python :: 3.9',
      'Programming Language :: Python :: 3.10',
      'Programming Language :: Python :: 3.11'
    ]
)

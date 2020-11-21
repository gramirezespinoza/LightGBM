import os
import shutil
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

# TODO fix this shit

# clean previous build
# from: https://stackoverflow.com/questions/16993927/using-cython-to-link-python-to-a-shared-library
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if name.endswith(".so"):
            os.remove(os.path.join(root, name))
    for name in dirs:
        if name == "build" and os.path.isdir(name):
            shutil.rmtree(name)

dataset_extension = Extension(
    name="dataset",
    sources=["dataset.pyx"],
    libraries=["_lightgbm"],
    include_dirs=["/Users/gire/Desktop/Repositories/LightGBM/include/"],
    extra_link_args=["-L./../"],
    extra_compile_args=["-std=c++14", "-mmacosx-version-min=10.15"],
    language="c++",
)

boosting_extension = Extension(
    name="boosting",
    sources=["boosting.pyx"],
    libraries=["_lightgbm"],
    include_dirs=["/Users/gire/Desktop/Repositories/LightGBM/include/"],
    extra_link_args=["-L./../"],
    extra_compile_args=["-std=c++14", "-mmacosx-version-min=10.15"],
    language="c++",
)

config_extension = Extension(
    name="config",
    sources=["config.pyx"],
    libraries=["_lightgbm"],
    include_dirs=["/Users/gire/Desktop/Repositories/LightGBM/include/"],
    extra_link_args=["-L./../"],
    extra_compile_args=["-std=c++14", "-mmacosx-version-min=10.15"],
    language="c++",
)

metric_extension = Extension(
    name="metric",
    sources=["metric.pyx"],
    libraries=["_lightgbm"],
    include_dirs=["/Users/gire/Desktop/Repositories/LightGBM/include/"],
    extra_link_args=["-L./../"],
    extra_compile_args=["-std=c++14", "-mmacosx-version-min=10.15"],
    language="c++",
)

objective_extension = Extension(
    name="objective",
    sources=["objective.pyx"],
    libraries=["_lightgbm"],
    include_dirs=["/Users/gire/Desktop/Repositories/LightGBM/include/"],
    extra_link_args=["-L./../"],
    extra_compile_args=["-std=c++14", "-mmacosx-version-min=10.15"],
    language="c++",
)

setup(
    name="lightgbm-cython",
    ext_modules=cythonize(
        [dataset_extension, boosting_extension, config_extension, metric_extension, objective_extension],
        language_level="3",
        build_dir="build")
)

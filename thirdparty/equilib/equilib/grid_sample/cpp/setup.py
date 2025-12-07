#!/usr/bin/env python3

from setuptools import Extension, setup

from torch.utils import cpp_extension


setup(
    name="grid_sample_cpp",
    ext_modules=[
        cpp_extension.CppExtension("grid_sample", ["grid_sample.cpp"])
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)

Extension(
    name="grid_sample_cpp",
    sources=["grid_sample.cpp"],
    include_dirs=cpp_extension.include_paths(),
    language="c++",
)

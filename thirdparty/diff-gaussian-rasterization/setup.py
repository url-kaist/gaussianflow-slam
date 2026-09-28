#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                # "cxx": ["-DTORCH_USE_CUDA_DSA=1"],
                "cxx": ['-O3', "-std=c++17"], # , "-g", "-O0"
                # "cxx": ["-DTORCH_USE_CUDA_DSA=1", "-g", "-O0"], # , "-g", "-O0"
                "nvcc": [
                    '-O3',
                    "-std=c++17", 
                    # "-DTORCH_USE_CUDA_DSA=1",
                    # "-G",  # Enables debug info for CUDA
                    # "-g",  # Symbol info for debugger
                    # "-O0", # No optimization (for better debugging)
                    # "-O2",
                    # "-lineinfo",  # include line-to-PTX mapping info
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
                    ]
                })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

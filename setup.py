from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='gfslam_backends',
    ext_modules=[
        CUDAExtension('gfslam_backends',
            include_dirs=[osp.join(ROOT, 'thirdparty/lietorch/eigen')],
            sources=[
                'src/gfslam.cpp',
                'src/gfslam_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
                'src/flow_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)

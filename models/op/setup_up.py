from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='upfirdn2d_cpp',
    ext_modules=[        
        CppExtension('upfirdn2d_cpp', ['upfirdn2d.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

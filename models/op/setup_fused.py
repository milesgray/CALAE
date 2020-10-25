from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='fused_bias_act_cpp',
    ext_modules=[
        CppExtension('fused_bias_act_cpp', ['fused_bias_act.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

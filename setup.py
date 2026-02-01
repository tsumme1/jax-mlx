import os
import sys
import subprocess
import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = 'Debug' if self.debug else 'Release'
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPython3_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
        ]

        build_args = ['--config', cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '-j'] + build_args, cwd=self.build_temp)
        
        # Copy the built library to the package source directory as well
        src_package_dir = os.path.join(ext.sourcedir, 'src', 'jax_mlx')
        built_lib = os.path.join(extdir, 'libmlx_pjrt_plugin.dylib')
        if os.path.exists(built_lib) and os.path.isdir(src_package_dir):
            shutil.copy2(built_lib, src_package_dir)

setup(
    name='jax-mlx',
    version='0.0.1',
    author='Thomas Summe',
    description='JAX PJRT plugin for Apple Silicon using MLX',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tsumme1/jax-mlx',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[CMakeExtension('jax_mlx.mlx_pjrt_plugin')],
    cmdclass=dict(build_ext=CMakeBuild),
    package_data={'jax_mlx': ['*.dylib', '*.so']},
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.11',
    install_requires=[
        'jax>=0.5.0',
        'jaxlib>=0.5.0',
        'mlx',
    ],
    entry_points={
        "jax_plugins": [
            "mlx_plugin = jax_mlx.plugin",
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

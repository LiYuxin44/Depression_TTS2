from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from pathlib import Path

here = Path(__file__).parent

exts = [
    Extension(
        name="core",  # 强制本地模块名，避免写入包路径
        sources=[str(here / "core.pyx")],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="monotonic_align",
    ext_modules=cythonize(exts, language_level=3),
)
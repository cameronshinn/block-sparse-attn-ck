import os, sys
import torch

try:
    from skbuild import setup
except ImportError:
    print('scikit-build is required to build from source.', file=sys.stderr)
    print('Please run:', file=sys.stderr)
    print('', file=sys.stderr)
    print('  python -m pip install scikit-build')
    sys.exit(1)

torch_root = os.path.dirname(torch.__file__)

setup(
    name="bsp-attn",  # This is the name you see under `pip list`
    version="0.0.1",
    description="block-sparse multi-head attention layer for PyTorch",
    license="MIT",
    packages=['bsp_attn'],  # These are the names you import the package as, and must match folder names?
    cmake_args=[f'-DCMAKE_PREFIX_PATH={torch_root}']
)

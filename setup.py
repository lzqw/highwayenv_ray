# Please don't change the order of following packages!
import sys
from distutils.core import setup

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

setup(
    name="highwayray",
    install_requires=[
    "ray[all]==2.9.0",
    "gymnasium==0.28.1",
    "tensorboardX==2.6.2.2",
    "highway-env",
    "chardet",
    "utils"
    ],
    license="Apache 2.0",
)

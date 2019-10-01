# Copyright 2019 Stanford
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import sys
import sysconfig
from setuptools import find_packages

# need to use distutils.core for correct placement of cython dll           
if "--inplace" in sys.argv:                                                
    from distutils.core import setup
    from distutils.extension import Extension                              
else:
    from setuptools import setup
    from setuptools.extension import Extension

def config_cython():
    sys_cflags = sysconfig.get_config_var("CFLAGS")
    try:
        from Cython.Build import cythonize
        ret = []
        path = "taso/_cython"
        for fn in os.listdir(path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(Extension(
                "taso.%s" % fn[:-4],
                ["%s/%s" % (path, fn)],
                include_dirs=["../include", "/usr/local/cuda/include"],
                libraries=["taso_runtime"],
                extra_compile_args=["-DUSE_CUDNN", "-std=c++11"],
                extra_link_args=[],
                language="c++"))
        return cythonize(ret, compiler_directives={"language_level" : 3})
    except ImportError:
        print("WARNING: cython is not installed!!!")
        return []

setup_args = {}

#if not os.getenv('CONDA_BUILD'):
#    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
#    for i, path in enumerate(LIB_LIST):
#    LIB_LIST[i] = os.path.relpath(path, curr_path)
#    setup_args = {
#        "include_package_data": True,
#        "data_files": [('taso', LIB_LIST)]
#    }

setup(name='taso',
      version="0.1.0",
      description="TASO: A Tensor Algebra SuperOptimizer for Deep Learning",
      zip_safe=False,
      install_requires=[],
      packages=find_packages(),
      url='https://github.com/jiazhihao/taso',
      ext_modules=config_cython(),
      #**setup_args,
      )


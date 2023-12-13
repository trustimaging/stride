
import re
import os
import sys


def go():
    # Get CUDA compilers path
    comp_path = None

    # Try to get from environment variables
    envs = ['CUDA_HOME', 'NVHPC_CUDA_HOME', 'CUDA_ROOT']
    for env in envs:
        cuda_home = os.environ.get(env, None)
        if cuda_home:
            comp_path = os.path.join(os.path.join(os.path.dirname(cuda_home), 'compilers'), 'lib')
            break

    if comp_path is None:
        hpcsdk_home = os.environ.get('HPCSDK_HOME')
        if hpcsdk_home:
            comp_path = os.path.join(os.path.join(hpcsdk_home, 'compilers'), 'lib')

    # If not, try to get from LD_LIBRARY_PATH
    if comp_path is None:
        library_path = os.environ.get('LD_LIBRARY_PATH', '')
        for path in library_path.split(':'):
            if re.match('.*/nvidia/hpc_sdk/.*/?compilers/lib', path) or \
                    re.match('.*/nvhpc/.*/?compilers/lib', path):
                comp_path = path

    if comp_path is None or not os.path.exists(comp_path):
        return ''

    # Now try to get libgomp path
    libnames = ['libgomp', 'libiomp', 'libomp']
    extnames = ['.so', '.dylib', '.dll']
    for lib in libnames:
        for ext in extnames:
            lib_path = os.path.join(comp_path, lib + ext)
            if os.path.exists(lib_path):
                return lib_path

    return ''


if __name__ == '__main__':
    sys.exit(go())

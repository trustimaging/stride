#!/usr/bin/env bash
#
# Build a DOLFINx environment for the stride FEM problem types, from scratch.
#
#   ./scripts/create_fem_env.sh [complex|real] [env-name]
#
# Defaults to the complex build, named stride-dolfinx-<scalar>.
#
# Why a script rather than four commands in the README. Every failure this has produced so far
# came from the steps being run apart: a shell variable that was not set, a base environment
# that never got applied, a "conda install python=3.11" that silently swapped the complex PETSc
# for the real one. The constraints only hold if they are in place before anything is installed
# and nothing re-solves afterwards, which is easier to guarantee in one pass than by hand.
#
# What it does, in order:
#
#   1. exports the existing environment, if there is one, so a failed rebuild is recoverable
#   2. removes it
#   3. creates it empty and writes conda-meta/pinned BEFORE any package is installed
#   4. applies environment.yml, then the FEM overlay
#   5. checks the Python version, the scalar type and the hdf5 pairing, and stops if any is wrong
#   6. installs stride and stride-private editable, with --no-deps
#
# The pins matter for different reasons. MPI has to be pinned before the base environment goes
# in, because on Linux the base pulls mpich through hdf5 and the overlay's openmpi pin then
# cannot be satisfied. The PETSc scalar type has to be pinned because a build-string constraint
# living only in a yml is forgotten by the next solve, and conda-forge builds real by default.
# hdf5 has to be pinned because fenics-dolfinx 0.9.0 holds it at 1.14 while an unconstrained
# h5py is built against a newer one, and the mismatch surfaces as "Not a datatype" on import.

set -euo pipefail

SCALAR="${1:-complex}"
ENV_NAME="${2:-stride-dolfinx-$SCALAR}"

STRIDE="${STRIDE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STRIDE_PRIVATE="${STRIDE_PRIVATE:-$(dirname "$STRIDE")/stride-private}"

case "$SCALAR" in
    real)    expected_scalar="float64" ;;
    complex) expected_scalar="complex128" ;;
    *) echo "Error: scalar must be real or complex, got '$SCALAR'" >&2; exit 1 ;;
esac

OVERLAY="$STRIDE/environment-fem-$SCALAR.yml"
BASE="$STRIDE/environment.yml"

for file in "$BASE" "$OVERLAY"; do
    [ -f "$file" ] || { echo "Error: $file not found" >&2; exit 1; }
done

[ -d "$STRIDE_PRIVATE" ] || echo "Note: no stride-private at $STRIDE_PRIVATE, skipping it"

echo "==> building $ENV_NAME ($SCALAR) from $BASE + $OVERLAY"

# 1. a rebuild that dies halfway through, on a network hiccup say, should not leave you with
#    nothing. The export is not a substitute for the ymls, it is a record of what was there
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    BACKUP="$HOME/$ENV_NAME-$(date +%Y%m%d-%H%M%S).yml"
    echo "==> exporting the current $ENV_NAME to $BACKUP"
    conda env export -n "$ENV_NAME" > "$BACKUP"

    echo "==> removing $ENV_NAME"
    conda env remove -y -n "$ENV_NAME"
fi

# 2. empty first, so the pins are in place before the solver ever runs
echo "==> creating $ENV_NAME and pinning"
conda create -y -n "$ENV_NAME"

PREFIX="$(conda info --base)/envs/$ENV_NAME"
printf '%s\n' \
    'mpi=*=*openmpi*' \
    "petsc=*=*$SCALAR*" \
    'hdf5=1.14.*' \
    > "$PREFIX/conda-meta/pinned"

echo "    $(tr '\n' ' ' < "$PREFIX/conda-meta/pinned")"

# 3. base then overlay. The overlay is additive on purpose, so both are needed
echo "==> applying $BASE"
conda env update -n "$ENV_NAME" -f "$BASE"

echo "==> applying $OVERLAY"
conda env update -n "$ENV_NAME" -f "$OVERLAY"

# 4. check before installing anything on top, because every one of these has failed silently
echo "==> checking the environment"
conda run --no-capture-output -n "$ENV_NAME" python - "$expected_scalar" <<'PYTHON'
import sys

expected = sys.argv[1]
problems = []

major, minor = sys.version_info[:2]
print('python  %d.%d' % (major, minor))

# mosaic's _profile.c has branches for 3.10 and 3.11 only, and falls through to code using
# f_stacktop, which was removed in 3.11, so pip install -e . cannot compile above that
if (major, minor) not in ((3, 10), (3, 11)):
    problems.append('python is %d.%d, and mosaic only compiles on 3.10 or 3.11. The base '
                    'environment pins this, so a wrong version means it never applied'
                    % (major, minor))

import numpy
import dolfinx
import petsc4py.PETSc as PETSc

# petsc4py and DOLFINx are separately compiled against PETSc and can disagree. It is the
# DOLFINx one the operators gate on, so both have to be right
scalars = (numpy.dtype(PETSc.ScalarType).name, numpy.dtype(dolfinx.default_scalar_type).name)
print('scalar  petsc4py=%s dolfinx=%s' % scalars)

if scalars != (expected, expected):
    problems.append('expected %s, and only the petsc and slepc packages carry the scalar type '
                    'in their build string -- check conda list | grep petsc' % expected)

import h5py

built, running = h5py.version.hdf5_built_version_tuple, h5py.version.hdf5_version_tuple
print('hdf5    h5py built against %s, running against %s' % (built, running))

if built != running:
    problems.append('h5py was built against a different hdf5 than the one installed, which '
                    'fails on import as "Not a datatype"')

from mpi4py import MPI

print('mpi     %s' % MPI.Get_library_version().splitlines()[0].strip())

# mixing MPI implementations between the base environment and the overlay hangs at exit,
# inside MPI_Finalize, long after everything has otherwise worked
if 'Open MPI' not in MPI.Get_library_version():
    problems.append('this is not Open MPI, and mixing implementations hangs at exit')

if problems:
    print('\nFAILED:')
    for problem in problems:
        print('  - %s' % problem)
    raise SystemExit(1)

print('\nenvironment OK')
PYTHON

# 5. --no-deps because the conda environment already has everything setup.py asks for. Letting
#    pip re-resolve it pulls PyPI wheels in over the conda packages -- torch and the CUDA stack
#    among them -- which is where the ABI mismatches start
echo "==> installing stride editable"
conda run --no-capture-output -n "$ENV_NAME" pip install -e "$STRIDE" --no-deps

if [ -d "$STRIDE_PRIVATE" ]; then
    echo "==> installing stride-private editable"
    conda run --no-capture-output -n "$ENV_NAME" pip install -e "$STRIDE_PRIVATE" --no-deps
fi

echo "==> running the meshed tests"
conda run --no-capture-output -n "$ENV_NAME" python -m pytest "$STRIDE/stride/tests" -q

if [ -d "$STRIDE_PRIVATE/stride_private/tests" ]; then
    conda run --no-capture-output -n "$ENV_NAME" \
        python -m pytest "$STRIDE_PRIVATE/stride_private/tests" -q
fi

echo
echo "$ENV_NAME is ready:  conda activate $ENV_NAME"

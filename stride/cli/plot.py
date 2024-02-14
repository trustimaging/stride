
import click

from stride import plotting
from stride.problem.data import *
from stride.problem import Acquisitions, Geometry


def _intantiate(filename):
    try:
        return ScalarField(name=filename)
    except AttributeError:
        pass
    try:
        return VectorField(name=filename)
    except AttributeError:
        pass
    try:
        return Traces(name=filename)
    except AttributeError:
        pass
    try:
        return SparseField(name=filename)
    except AttributeError:
        pass
    try:
        return SparseCoordinates(name=filename)
    except AttributeError:
        pass
    try:
        return Acquisitions(name=filename)
    except AttributeError:
        pass
    try:
        return Geometry(name=filename)
    except AttributeError:
        pass


@click.command()
@click.argument('filenames', required=True, nargs=-1)
@click.version_option()
def go(filenames=None, **kwargs):
    axis = None
    for filename in filenames:
        field = _intantiate(filename)
        field.load(filename=filename)

        axis = field.plot(axis=axis, plot=False)

    plotting.show(axis)


if __name__ == '__main__':
    go()

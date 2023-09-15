import os
from stride.utils.fullwave import *
from stride import Time, Space, Problem, ScalarField
import click


@click.command()
# path and project name
@click.option('--path', type=str, required=True,
              prompt='path to folder with fullwave data',
              help='''path of your fullwave files (they all have to be in the
                      same path)''')
@click.option('--prjname', type=str, required=True,
              prompt='name of the project (to prepend to Stride files)',
              help='prefix name for Stride files')
# acquisition data
@click.option('--ttrname', type=str, required=True,
              prompt='observed data ttr file name',
              help='observed data ttr file to convert')
@click.option('--ttr0000', type=bool, required=False, default=False,
              prompt='do you have a 0000.ttr file?',
              help='''whether the ttr data file is empty (0000.ttr file),
              '(does not matter if you do not have a data ttr file)''')
# geometry
@click.option('--srcpgyname', type=str, required=True,
              prompt='source pgy file name',
              help='source pgy file name')
@click.option('--recpgyname', type=str, required=False, default='null',
              prompt='receiver pgy file name',
              help='receiver pgy file name (None if you do not have one)')
# source signature
@click.option('--srcttrname', type=str, required=False, default='null',
              prompt='source ttr file name',
              help='''source ttr file to convert to Stride (None if you do not have one or
              do not want to convert it)''')
# vp model
@click.option('--vpvtrname', type=str, required=False, default='null',
              prompt='vp model vtr file name',
              help='vp model vtr file to convert to Stride format (None if you do not have one)')
@click.option('--vpvalue', type=str, required=False, default='null',
              prompt='constant vp value to generate model',
              help='constant vp value to populate model in m/s (if vp vtr provided will ignore this value)')
# other
@click.option('--writedata', type=bool, required=False, default=True,
              prompt='write data read from Time.ttr file?',
              help='write read data from Time.ttr or not')
@click.option('--srcrecsplit', type=bool, required=False, default=True,
              prompt='are src pgy and rec pgy different?',
              help='''whether srcs and recs are different in the pgy files so that they can be treated
              as separate transducers (if not, only src pgy is read and the same transducer positions are
              used for the receivers)''')
@click.option('--dx', type=float, required=True,
              prompt='model dx',
              help='model spacing in meters (same for all three dimensions)')
@click.option('--extra', type=int, required=True,
              prompt='extra cells to pad model',
              help='extra cells to add to the edges of the model (same for all dimensions')
@click.option('--absorb', type=int, required=True,
              prompt='absorbing cells',
              help='absorbing cells in the extra cells of the model (most outer ones)')
@click.option('--plot', type=bool, required=False, default=False,
              prompt='plot data and src traces?',
              help='plot traces and sources')
def go(**kwargs):
    path = kwargs.pop('path')
    prjname = kwargs.pop('prjname')
    ttrname = kwargs.pop('ttrname')
    ttr0000 = kwargs.pop('ttr0000')
    srcpgyname = kwargs.pop('srcpgyname')
    recpgyname = kwargs.pop('recpgyname')
    if recpgyname == 'null':
        recpgyname = None
    srcttrname = kwargs.pop('srcttrname')
    if srcttrname == 'null':
        srcttrname = None
    vpvtrname = kwargs.pop('vpvtrname')
    if vpvtrname == 'null':
        vpvtrname = None
    vpvalue = kwargs.pop('vpvalue')
    if vpvalue == 'null':
        vpvalue = None
    else:
        vpvalue = float(vpvalue)
    writedata = kwargs.pop('writedata')
    srcrecsplit = kwargs.pop('srcrecsplit')
    dx = kwargs.pop('dx')
    extra = kwargs.pop('extra')
    absorb = kwargs.pop('absorb')
    plot = kwargs.pop('plot')

    # Create the temporal grid
    _, _, num, stop, dt = read_header_ttr(os.path.join(path, ttrname))
    start = 0.
    time = Time(start=start,
                step=dt,
                num=num)

    # Create the spatial grid
    src_shape = read_header_pgy(os.path.join(path, srcpgyname))

    if recpgyname is not None:
        rec_shape = read_header_pgy(os.path.join(path, recpgyname))
        assert src_shape == rec_shape, 'model dimensions mismatch in src and rec pgys'

    space = Space(shape=(src_shape[0], src_shape[1], src_shape[2]),
                  extra=(extra, extra, extra),
                  absorbing=(absorb, absorb, absorb),
                  spacing=(dx, dx, dx))

    # Create problem
    problem = Problem(name=prjname, space=space, time=time)

    # Create vp ScalarField
    if vpvtrname is not None:
        vp = ScalarField(name='vp', grid=problem.grid)
        vp.data[:] = read_vtr_model3D(vtr_path=os.path.join(path, vpvtrname))
        vp.pad()
    else:
        vp = ScalarField(name='vp', grid=problem.grid)
        vp.fill(vpvalue)

    # Create medium
    problem.medium.add(vp)

    # Create point transducers
    problem.transducers.default()

    if recpgyname is not None:
        recpgyname = os.path.join(path, recpgyname)

    offset_id = problem.geometry.from_fullwave(os.path.join(path, srcpgyname), recpgyname)

    print('---parameters')
    print('offset id:  ', offset_id)
    print('num: {:f}'.format(num))
    print('dt: {:f}'.format(dt))
    print('stop: {:f}'.format(stop))
    print('shape: ', src_shape)
    print('dx: {:f}'.format(dx))
    print('extra: ', extra)
    print('absorb: ', absorb)
    print('num shots: ', problem.acquisitions.num_shots)

    # Load acquisition data
    if srcttrname is not None:
        srcttrname = os.path.join(path, srcttrname)

    problem.acquisitions.from_fullwave(
        acquisition_path=os.path.join(path, ttrname),
        source_path=srcttrname,
        read_traces=writedata, src_rcv_split=srcrecsplit, offset_id=offset_id)

    # Plot if required
    if plot:
        problem.plot()

    # Dump problem to disk
    problem.dump()

# TODO fix read_observed_ttr in fullwave.py to handle 0000.ttr properly
#
# if no optional args are given, raise an exception


if __name__ == '__main__':
    go()

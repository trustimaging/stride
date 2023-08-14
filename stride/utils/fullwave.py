
from collections import defaultdict
import numpy as np
import struct
from types import NoneType

import mosaic


__all__ = ['read_vtr_model3D', 'read_observed_ttr', 'read_signature_ttr',
           'read_signature_txt', 'read_geometry_pgy']


def read_vtr_model3D(vtr_path, swap_axes=False):
    """
    Function to read 3D vtr low kernel binary data (model files) and
    returns the data as an as ndarray.

    Code adapted from Oscar Calderon from Imperial College London's
    FULLWAVE consortium.

    Parameters
    ----------
    vtr_path : str
        The path to the vtr file, which contains the 3D model
    swap_axes : bool, optional
        Permutes Fullwave storing format (depth, cross-line, in-line) to stride format (in-line,
        depth, cross-line). Default False.

    Returns
    -------
    model : ndarray
        File that was read from vtr
    """
    with open(vtr_path, 'rb') as f:
        # Read headers from binary file.
        rec_len = np.fromfile(f, dtype='int32', count=1)
        ncomp = np.fromfile(f, dtype='int32', count=1)
        ndim = np.fromfile(f, dtype='int32', count=1)
        extranum = np.fromfile(f, dtype='int32', count=1)

        rec_len = np.fromfile(f, dtype='int32', count=1)
        rec_len = np.fromfile(f, dtype='int32', count=1)
        n3 = np.fromfile(f, dtype='int32', count=1)
        n2 = np.fromfile(f, dtype='int32', count=1)
        n1 = np.fromfile(f, dtype='int32', count=1)
        rec_len = np.fromfile(f, dtype='int32', count=1)

        # Read model:
        ind1 = 0
        ind2 = -1
        model = np.zeros((int(n1), int(n2), int(n3)))
        trace = np.zeros((int(n3), int(1)))
        while True:
            ind2 = ind2 + 1
            if ind2 > n2-1:
                ind2 = 0
                ind1 = ind1 + 1
                if ind1 > n1-1:
                    break

            rec_len = np.fromfile(f, dtype='int32', count=1)
            if rec_len.size < 1:
                break
            else:
                trace = np.fromfile(f, dtype='f4', count=int(n3))
                rec_len = np.fromfile(f, dtype='int32', count=1)
                model[ind1, ind2, :] = trace[:]
    if swap_axes:
        model = np.transpose(model, (2, 1, 0))
    return model


def read_header_ttr(ttr_path):
    """
    Function to read ttr header.
    Parameters
    ----------
    ttr_path : str
        The path to the ttr file, which contains metadata about model geomtery.

    Returns
    -------
    num_composite_shots : int
        The number of shots made during the experiment.
    max_num_rec_per_src : int
        The maximum number of receivers seen by a source.
    num_samples : int
        The total number of samples in the time domain.
    total_time : float
        The acquisition time per shot in seconds.
    """

    # Read header
    # header structure: _ (int), num_shots (int), num_recs_max (int), num_steps (int), total_time (float), _ (int)
    with open(ttr_path, mode='rb') as file:
        nheader = 1 + 4 + 1  # number of variables in header with trailing integers
        headers = file.read(4 * nheader)
        headers = struct.unpack('iiiifi', headers)
        _, num_composite_shots, max_num_rec_per_src, num_samples, total_time, _ = headers

    # TODO Calculate dt here

    return num_composite_shots, max_num_rec_per_src, num_samples, total_time


def read_observed_ttr(ttr_path, store_traces=True, has_traces=True):
    """
    Function to read acquisition parameters and data from Fullwave's Observed.ttr binary
    file. Adapted from code provided by Oscar Calderon from Imperial College London's FULLWAVE
    consortium.

    Parameters
    ----------
    ttr_path: str
        Path to ttr file
    store_traces: bool, optional
        Flag to store the data inside the ttr. If true, data is saved in memory and returned
        by function. If False, only the source and receiver IDs are returned. Default False
    has_traces : bool, optional
        Flag that determines if ttr file contains any data. Default True.

    Returns
    -------
    observed: dict
    """

    # structure: observed[shot_id][rec_id] = trace
    observed = defaultdict(lambda: defaultdict(NoneType))

    with open(ttr_path, mode='rb') as file:  # get traces and populate observed

        # Read header
        # header structure: _ (int), num_shots (int), num_recs_max (int), num_steps (int), total_time (float), _ (int)

        nheader = 1 + 4 + 1  # number of variables in header with trailing integers
        headers = file.read(4 * nheader)
        headers = struct.unpack('iiiifi', headers)
        _, num_shots, num_rec_max, num_steps, total_time, _ = headers

        # Read rows
        # row structure: _ (int), shot_id (int), rec_id (int), range[1, num_steps] (float),  _ (int)

        num_elements_per_row = 1 + 2 + num_steps + 1  # number of variables in row with trailing integers

        trace_counter = 0
        while True:
            trace_counter += 1
            row = file.read(4*num_elements_per_row)

            if not row:
                break  # End of file
            try:
                if not has_traces:
                    row = struct.unpack('<iii' + num_steps*'f' + 'i', row)  # read assuming 0000_ttr
                    trace_array = None
                else:
                    row = struct.unpack('<iii' + num_steps*'f' + 'i', row)  # read assuming observed_ttr
                    trace_array = np.array(row[3:-1], dtype=np.float32)

                shot_id = row[1] - 1  # Fullwave starts count from 1, stride from 0
                rec_id = row[2] - 1

                if not store_traces:
                    # create empty observed dict
                    observed[shot_id][rec_id] = None  # populate with blanks

                elif store_traces and has_traces:
                    observed[shot_id][rec_id] = trace_array

                elif store_traces and not has_traces:
                    raise Exception('If ttr does not contain data, '
                                    'then observed data cannot be stored. '
                                    'Try exist_traces=False.')  # for exist=False adn store=True

            except struct.error as e:
                mosaic.logger().warn("Warning: Line %g of %s file could "
                                     "not be unpacked" % (trace_counter, ttr_path.split("/")[-1]))

    return observed


def read_signature_ttr(ttr_path):
    """
    Reads ttr signature data from Fullwave's Signature.ttr binary
    file. Adapted from code provided by Oscar Calderon from Imperial College London's FULLWAVE
    consortium.

    Parameters
    ----------
    ttr_path: str
        Path to ttr file

    Returns
    -------
    wavelets: list
        List of wavelet for every source ID
     """

    # Dict for traces
    wavelets = {}

    with open(ttr_path, mode='rb') as file:

        # Read header
        # header structure: _ (int), num_shots (int), num_max_pntShot_per_csShot (int),
        # num_steps (int), total_time (float), _ (int)

        nheader = 1 + 4 + 1  # number of variables in header with trailing integers
        headers = file.read(4 * nheader)
        headers = struct.unpack('iiiifi', headers)
        _, nsrc, maxptsrc, nt, ttime, _ = headers

        num_wavelets = nsrc  # NOTE assumes num_max_pntShot_per_csShot = 1

        # Read rows
        # row structure: _ (int), shot_id (int), rec_id (int), range[1, num_steps] (float),  _ (int)

        num_row = 1 + 2 + nt + 1  # number of variables in row with trailing integers

        for i in range(num_wavelets):  # csref is a composite,
            row = file.read(4*num_row)

            if not row:
                break  # End of file

            row = struct.unpack('<iii' + nt*'f' + 'i', row)

            composite_shot_id = row[1] - 1    # Fullwave starts count from 1, stride from 0
            point_source_id = row[2] - 1   # Fullwave starts count from 1, stride from 0

            wavelet_array = np.array(row[3:-1], dtype=np.float32)

            wavelets[composite_shot_id] = wavelet_array

    return wavelets


def read_signature_txt(txt_path):
    """
    Reads .txt signature data from Fullwave's Signature.txt file.
    Adapted from code provided by Oscar Calderon from Imperial College London's FULLWAVE
    consortium.

    Parameters
    ----------
    txt_path: str
        Path to txt file

    Returns
    -------
    wavelet: list
        List of the wavelet data as read from txt_path
     """
    with open(txt_path, "r+") as f:
        wavelet = f.read().splitlines()
    wavelet = [float(w) for w in wavelet]
    return wavelet


def read_geometry_pgy(geom_path, **kwargs):
    """
    Function to read geometry coordinates and data from Fullwave's .pgy file.
    Adapted from code provided by Oscar Calderon from Imperial College London's FULLWAVE
    consortium.

    Parameters
    ----------
    geom_path: str
        Path to Fullwave pgy geometry file
    scale: float, optional
        Value to each scale the location values in all dimensions. Useful for unit conversion. Default 1.
    disp: tuple or float, optional
        Amount to displace in each dimension in Metres. Applied after scale and before swapaxes. Default (0., 0., 0.)
    drop_dims: tuple or int, optional
        Coordinate dimensions of pgy file to drop (count from 0). Default ().
    swap_axes: bool, optional
        Permutes Fullwave storing format (depth, cross-line, in-line) to stride format (in-line, depth,
        cross-line). Default False.

    Returns
    -------
    ids: ndarray
        ID number of every transducer
    coordinates:
        Coordinate of each transducer with format (num_transducers, x, y, z). Coordinates will match
        format specified by <swap_axes>

    """
    num_locations, n3, n2, n1 = -1, -1, -1, -1
    scale = kwargs.get('scale', (1., 1., 1.))
    disp = kwargs.get('disp', (0., 0., 0.))
    drop_dims = kwargs.get('drop_dims', ())
    swap_axes = kwargs.get('swap_axes', True)

    # Read coordinates and IDs from pgy file
    with open(geom_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.split()

            # Header line
            if i == 0:
                # nz (depth), ny (cross-line), nx (inline) in fullwave format
                num_locations, n3, n2, n1 = [int(h) for h in line]
                dim = len(line)-1
                coordinates = np.zeros((num_locations, dim))
                ids = np.zeros((num_locations), dtype=int)

                # Reformating displacement array to match coordinate dimensions
                try:
                    disp = list(disp)
                    while len(disp) < dim:
                        disp.append(0.)
                except TypeError:  # if disp variable does not have length
                    disp = list(np.full((dim, ), disp))

            # Transducer IDs and Coordinates
            else:
                ids[i-1] = int(line[0]) - 1     # Fullwave starts count from 1, stride from 0
                _coordinates = [scale[i]*(float(c)-1) + float(disp[i]) for i, c in enumerate(line[1:])]
                if swap_axes:
                    _coordinates = [_coordinates[nx] for nx in (2, 1, 0)]
                coordinates[i-1] = _coordinates
    assert len(coordinates) == len(ids) == num_locations

    # Drop dimensions if prompted
    coordinates = np.delete(coordinates, obj=drop_dims, axis=1)

    return ids, coordinates

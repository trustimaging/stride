
import numpy as np
import struct
import mosaic


__all__ = ['read_vtr_model3D', 'read_observed_ttr', 'read_signature_ttr',
            'read_signature_txt', 'read_geometry_pgy']


def read_vtr_model3D(vtr_path, swap_axes=False):
    '''
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
    '''
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

    # Read 4-byte binary ttr file to retrieve source ids and correspondent receiver ids
    with open(ttr_path, mode='rb') as file:

        # List to store source and receivers ids
        sources_ids, receiver_ids = [], []
        shottraces = []
        tmp_receiver_ids, tmp_traces = [], []

        # Read header
        nheader = 1 + 4 + 1  # number of variables in header with trailing integers
        headers = file.read(4 * nheader)
        headers = struct.unpack('iiiifi', headers)
        _, num_composite_shots, max_num_rec_per_src, num_samples, total_time, _ = headers
    return num_composite_shots, max_num_rec_per_src, num_samples, total_time


def read_observed_ttr(ttr_path, store_traces=True):
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

    Returns
    -------
    sources_ids: list
        List of sources ID numbers
    receiver_ids: list
        Nested lists containing receiver IDs for each source ID as read from ttr file
    shottraces: list
        Nested lists containing the trace for each receiver ID of every source ID as
        read from ttr file. Will be empty if store_traces is False.
    """
    # Read 4-byte binary ttr file to retrieve source ids and correspondent receiver ids
    with open(ttr_path, mode='rb') as file:

        # List to store source and receivers ids
        sources_ids, receiver_ids = [], []
        shottraces = []
        tmp_receiver_ids, tmp_traces = [], []

        # Read header
        nheader = 1 + 4 + 1  # number of variables in header with trailing integers
        headers = file.read(4 * nheader)
        headers = struct.unpack('iiiifi', headers)
        _, ncomp, maxrecnum, nt, ttime, _ = headers

        # Read rows
        nrow = 1 + 2 + nt + 1  # number of variables in row with trailing integers

        while True:
            row = file.read(4*nrow)

            if not row:
                break  # End of file
            try:
                row = struct.unpack('<iii' + nt*'f' + 'i', row)

                csref = row[1] - 1    # Fullwave starts count from 1, stride from 0
                rcvref = row[2] - 1   # Fullwave starts count from 1, stride from 0
                trace = np.array(row[3:-1], dtype=np.float32)

                # Append shot id
                sources_ids.append(csref)
                sources_uids = list(set(sources_ids)) # unique source ids only

            except struct.error as e:
                mosaic.logger().warn("Warning: Line %g of %s file could not be unpacked" % (cnt, ttr_path.split("/")[-1]))

    with open(ttr_path, mode='rb') as file:

        # Read header
        nheader = 1 + 4 + 1  # number of variables in header with trailing integers
        headers = file.read(4 * nheader)
        headers = struct.unpack('iiiifi', headers)
        _, ncomp, maxrecnum, nt, ttime, _ = headers

        shottraces = [[] for i in range(ncomp)]
        receiver_ids = [[] for i in range(ncomp)]
        cnt = 0
        while True:
            cnt += 1
            row = file.read(4*nrow)

            if not row:
                break  # End of file
            try:
                row = struct.unpack('<iii' + nt*'f' + 'i', row)

                csref = row[1] - 1    # Fullwave starts count from 1, stride from 0
                rcvref = row[2] - 1   # Fullwave starts count from 1, stride from 0
                trace = np.array(row[3:-1], dtype=np.float32)

                idx_uid = sources_uids.index(csref)

                # Append all data from single csref id
                receiver_ids[idx_uid] = receiver_ids[idx_uid] + [rcvref]

                # Store traces to memory -- ! need to add option to save to file instead
                if store_traces:
                    shottraces[idx_uid] = shottraces[idx_uid] + [trace]

            except struct.error as e:
                mosaic.logger().warn("Warning: Line %g of %s file could not be unpacked" % (cnt, ttr_path.split("/")[-1]))

    return sources_uids, receiver_ids, shottraces


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
    with open(ttr_path, mode='rb') as file:
        # Read header
        nheader = 1 + 4 + 1  # number of variables in header with trailing integers
        headers = file.read(4 * nheader)
        headers = struct.unpack('iiiifi', headers)
        _, nsrc, maxptsrc, nt, ttime, _ = headers

        # Assert not composite source for code assumptions later on
        comp = True if maxptsrc > 1 else False
        if comp:
            raise NotImplementedError(" Files with composite sources are not yet supported inside Stride conversion. ")

        # Max number of wavelets is the total number of comp shots x max number of pt sources per comp shot
        nwavelets = nsrc * maxptsrc

        # List to store source ids and wavelets
        wavelets = np.empty((nwavelets, nt))

        # Read rows
        nrow = 1 + 2 + nt + 1  # number of variables in row with trailing integers

        for i in range(nwavelets):
            row = file.read(4*nrow)
            if not row:
                break  # End of file
            row = struct.unpack('<iii' + nt*'f' + 'i', row)
            csref = row[1] - 1    # Fullwave starts count from 1, stride from 0
            pntref = row[2] - 1   # Fullwave starts count from 1, stride from 0
            data = np.array(row[3:-1])
            wavelets[i] = data
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
                num_locations, n3, n2, n1 = [int(h) for h in line]  # nz (depth), ny (cross-line), nx (inline) in fullwave format
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

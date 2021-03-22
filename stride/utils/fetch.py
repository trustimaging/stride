

import os
import subprocess


__all__ = ['fetch']


known_assets = {
    'anastasio2D': 'https://api.github.com/repos/trustimaging/stride/releases/assets/32679235',
    'anastasio3D': 'https://api.github.com/repos/trustimaging/stride/releases/assets/32679251',
}


try:
    from stride_private.utils.fetch import known_assets
except ImportError:
    pass


def fetch(origin, dest, token=None):
    """
    Fetch asset from GitHub repository. The asset is only fetched once.

    Parameters
    ----------
    origin : str
        Name of the asset to fetch.
    dest : str
        Path to save file.
    token : str, optional
        Access token to fetch the access if it lives in a private repo.

    Returns
    -------

    """
    if os.path.exists(dest):
        return

    folder = os.path.dirname(dest)
    if not os.path.exists(folder):
        os.makedirs(folder)

    if origin in known_assets:
        origin = known_assets[origin]

    if token is not None:
        cmd = 'curl -LJ# -H "Authorization: token %s" -H ' \
              '"Accept: application/octet-stream" "%s" --output "%s"' % (token, origin, dest)
    else:
        cmd = 'curl -LJ# -H "Accept: application/octet-stream" "%s" --output "%s"' % (origin, dest)

    process = subprocess.run(cmd,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

    if process.returncode < 0:
        raise RuntimeError('Fetching with cmd "%s" failed' % cmd)

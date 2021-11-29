
import numpy as np


__all__ = ['add_noise']


def add_noise(data_0, target_snr_db):
    """
    Add noise to data to obtain target SNR (in dB).

    Parameters
    ----------
    data_0 : ndarray
        Data to which noise should be added.
    target_snr_db : float
        Target SNR in dB.

    Returns
    -------
    ndarray
        Data with added noise.

    """
    if target_snr_db > 0:
        data_0_watts = data_0 ** 2

        # Calculate signal power and convert to dB
        data_0_avg_watts = np.mean(data_0_watts)
        data_0_avg_db = 10 * np.log10(data_0_avg_watts)

        # Calculate noise then convert to watts
        noise_avg_db = data_0_avg_db - target_snr_db
        data_0_noise_avg_watts = 10 ** (noise_avg_db / 10)

        # Generate a sample of white noise
        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(data_0_noise_avg_watts), data_0.shape)
        data_0 = data_0 + noise_volts

    return data_0

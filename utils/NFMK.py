import numpy as np


def compute_moment(x, moment, axis, mean=None):
    if np.abs(moment - np.round(moment)) > 0:
        raise ValueError("All moment parameters must be integers")

    # moment of empty array is the same regardless of order
    if x.size == 0:
        return np.mean(x, axis=axis)

    dtype = x.dtype.type if x.dtype.kind in 'fc' else np.float64

    if moment == 0 or (moment == 1 and mean is None):
        # By definition the zeroth moment is always 1, and the first *central*
        # moment is 0.
        shape = list(x.shape)
        del shape[axis]

        if len(shape) == 0:
            return dtype(1.0 if moment == 0 else 0.0)
        else:
            return (np.ones(shape, dtype=dtype) if moment == 0
                    else np.zeros(shape, dtype=dtype))
    else:
        # Exponentiation by squares: form exponent sequence
        n_list = [moment]
        current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n - 1) / 2
            else:
                current_n /= 2
            n_list.append(current_n)

        # Starting point for exponentiation by squares
        mean = (x.mean(axis, keepdims=True) if mean is None
                else np.asarray(mean, dtype=dtype)[()])
        a_zero_mean = x - mean

        eps = np.finfo(a_zero_mean.dtype).resolution * 10
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.max(np.abs(a_zero_mean), axis=axis,
                              keepdims=True) / np.abs(mean)
        with np.errstate(invalid='ignore'):
            precision_loss = np.any(rel_diff < eps)
        n = x.shape[axis] if axis is not None else x.size
        if precision_loss and n > 1:
            print("Precision loss occurred in moment calculation due to "
                  "catastrophic cancellation. This occurs when the data "
                  "are nearly identical. Results may be unreliable.")

        if n_list[-1] == 1:
            s = a_zero_mean.copy()
        else:
            s = a_zero_mean ** 2

        # Perform multiplications
        for n in n_list[-2::-1]:
            s = s ** 2
            if n % 2:
                s *= a_zero_mean
        return np.mean(s, axis)


def kurtosis(a, axis=1, bias=True):
    n = a.shape[axis]
    mean = a.mean(axis, keepdims=True)
    m2 = compute_moment(a, 2, axis, mean=mean)
    m4 = compute_moment(a, 4, axis, mean=mean)
    with np.errstate(all='ignore'):
        zero = (m2 <= (np.finfo(m2.dtype).resolution * mean.squeeze()) ** 2)
        vals = np.where(zero, np.nan, m4 / m2 ** 2.0)

    if not bias:
        can_correct = ~zero & (n > 3)
        if can_correct.any():
            m2 = np.extract(can_correct, m2)
            m4 = np.extract(can_correct, m4)
            nval = 1.0 / (n - 2) / (n - 3) * ((n ** 2 - 1.0) * m4 / m2 ** 2.0 - 3 * (n - 1) ** 2.0)
            np.place(vals, can_correct, nval + 3.0)

    return vals


def mse_raw(x_raw: np.ndarray, x_estimated: np.ndarray):
    assert x_raw.shape == x_estimated.shape

    return np.average(np.square(x_raw - x_estimated))


def NFMK(x_raw: np.ndarray, x_estimated: np.ndarray, anomaly_ratio=0.05, lamb=0.1):
    assert x_raw.shape == x_estimated.shape

    # Compute the difference of true value and prediction value first
    diff = np.abs(x_raw - x_estimated)
    num_anomalies = int(anomaly_ratio * x_raw.shape[1])  # batch * length

    mask = np.ones(x_raw.shape, dtype=bool)
    for i in range(x_raw.shape[0]):
        anomaly_indices = np.argpartition(diff[i], -num_anomalies)[-num_anomalies:]
        mask[i][anomaly_indices] = False

    x_raw_removed = np.array([x_raw[i][mask[i]] for i in range(x_raw.shape[0])])
    x_estimated_removed = np.array([x_estimated[i][mask[i]] for i in range(x_estimated.shape[0])])
    # Minimize the kurtosis term
    mse_obj = mse_raw(x_raw_removed, x_estimated_removed)
    kurtosis_obj = 1 / np.mean(kurtosis(x_raw - x_estimated) - kurtosis(x_raw_removed - x_estimated_removed))

    return mse_obj + lamb * kurtosis_obj

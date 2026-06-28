import h5py
import hdf5plugin
import numpy as np
import math


def _extract_from_h5_by_index(filehandle, ev_start_idx: int, ev_end_idx: int, pixel_diff: int, roi):
    roi_h, roi_w = roi
    
    events = filehandle['events']
    x = events['x']
    y = events['y']
    t = events['t']

    x_new = (x[ev_start_idx:ev_end_idx] + pixel_diff).astype("int16")
    y_new = y[ev_start_idx:ev_end_idx].astype("int16")
    t_new = t[ev_start_idx:ev_end_idx].astype("int64")
    
    valid_mask = (x_new >= 0) & (x_new < roi_w) & (y_new >= 0) & (y_new < roi_h)

    x_new_filtered = x_new[valid_mask]
    y_new_filtered = y_new[valid_mask]
    t_new_filtered = t_new[valid_mask]

    output = {
        't': t_new_filtered,
        'x': x_new_filtered,  # Apply pixel difference to x coordinates
        'y': y_new_filtered,
    }
    return output




def extract_from_h5_by_timewindow(h5file, t_min_us_list, t_max_us_list, pixel_diff, roi):
    assert len(t_min_us_list) == len(t_max_us_list), "The lengths of t_min_us_list and t_max_us_list must be the same."

    with h5py.File(str(h5file), 'r') as h5f:
        ms2idx = np.asarray(h5f['ms_to_idx'], dtype='int64')
        events = h5f['events']
        t = events['t']

        results = []
        for t_ev_start_us, t_ev_end_us in zip(t_min_us_list, t_max_us_list):
            t_ev_start_ms = t_ev_start_us // 1000
            ms2idx_start_idx = t_ev_start_ms
            ev_start_idx = ms2idx[ms2idx_start_idx]

            assert t_ev_end_us <= t[-1], (t_ev_end_us, t[-1])
            t_ev_end_ms = math.floor(t_ev_end_us / 1000)
            ms2idx_end_idx = t_ev_end_ms
            ev_end_idx = ms2idx[ms2idx_end_idx]

            result = _extract_from_h5_by_index(h5f, ev_start_idx, ev_end_idx, pixel_diff, roi)
            results.append(result)

    return results

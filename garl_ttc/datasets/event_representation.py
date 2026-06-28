import numpy as np


def get_timevolume_roi_np(
        expand_box,
        x,
        y,
        tus,
        time_window=100e-3,
        number_of_planes=20,
        ):
    try:
        # Reference: Object Tracking by Jointly Exploiting Frame and Event Domain, ICCV 2021.
        tus = tus - tus[0]
        ts = tus * 1e-6
        tbin = time_window / number_of_planes

        xmin, ymin, xmax, ymax = map(int, expand_box)
        width = int(xmax - xmin)
        height = int(ymax - ymin)

        mask = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax) & (ts < time_window - 1e-5)

        x = x[mask].astype(np.int64) - xmin
        y = y[mask].astype(np.int64) - ymin
        ts = ts[mask].astype(np.float32)

        timevolume = np.zeros((number_of_planes, height, width), dtype=np.float32)
        evcount = np.zeros(number_of_planes, dtype=np.int32)
        if len(ts) == 0:
            return timevolume, evcount

        time_ind = (ts.astype(np.float64) / tbin).astype(np.int64)
        evcount = np.bincount(time_ind, minlength=number_of_planes).astype(np.int32)

        plane_size = height * width
        flat_idx = time_ind * plane_size + y * width + x
        order = np.argsort(flat_idx, kind='stable')
        sorted_flat = flat_idx[order]
        group_start = np.r_[0, np.flatnonzero(sorted_flat[1:] != sorted_flat[:-1]) + 1]
        group_count = np.diff(np.r_[group_start, len(sorted_flat)])

        last_pos = group_start + group_count - 1
        last_event = order[last_pos]
        prev_event = order[np.maximum(last_pos - 1, group_start)]

        plane_start = (sorted_flat[last_pos] // plane_size) * tbin
        prev_ts = np.where(group_count > 1, ts[prev_event], plane_start)
        values = np.exp(-((ts[last_event] - prev_ts) / tbin)).astype(np.float32)

        timevolume.reshape(-1)[sorted_flat[last_pos]] = values
    except Exception as exc:
        print(f'Error in get_timevolume_roi_np: {exc}')
        return None

    return timevolume, evcount

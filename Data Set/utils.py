import numpy as np

def to_polar(a):
    amplitude = np.linalg.norm(a)
    theta = np.arctan2(a[2], a[1])
    gamma = np.arctan2(a[1], a[0])

    return np.array([amplitude, theta, gamma])

def get_times_from_file(times_path):
    with open(times_path, 'r') as times_file:
        times = []
        start_time = 0

        # Get the times from each row of the times file
        for index, line in enumerate(times_file):
            if (not index == 0):
                times.append(float(line.split('\n')[0]))
            else:
                start_time = float(line.split('\n')[0])

        # Offset the times with the start time so the log file
        # and the times file will count the time in a similar way (from 00:00 in seconds)
        times = np.array(times)
        times += start_time

    return times
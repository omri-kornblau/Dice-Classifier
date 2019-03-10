import numpy as np
import utils

def get_times_from_file(times_file):
    times = []
    start_time = 0

    # Get the times from each row of the times file
    for index, line in enumerate(times_file):
        if (not index == 0):
            times.append(float(line.split('\n')[0])+start_time)
        else:
            start_time = float(line.split('\n')[0])

    # Offset the times with the start time so the log file
    # and the times file will count the time in a similar way (from 00:00 in seconds)
    times = np.array(times)

    return times

def get_data_from_line(log_line):
    """Get a string and return numpy array of numbers in it.

    The data is stored in the form: [time , accX, accY, accZ]
    Parameters
        log_line: string
            A line from the accelerations log file

    """
    raw_data = [""]

    for i, c in enumerate(log_line):
        if (c.isdigit()):
            # handle (-) (minus) sign
            if (len(raw_data[-1]) == 0 and log_line[i-1] == '-'):
                raw_data[-1] += '-'
            raw_data[-1] += c
        elif (log_line[i-1].isdigit()):
            raw_data[-1] = float(raw_data[-1])
            raw_data.append("")

    # Convert time to seconds
    time = raw_data[0]*360 + raw_data[1]*60 + raw_data[2] + raw_data[3]*0.001

    return (np.array([time, raw_data[4], raw_data[5], raw_data[6]]))

def get_data_from_file(log_file, lines_limit=0, to_polar=False):
    """Create ndarray object from accelerations log file.

    The data is stored inside given data set name in the object.
    """
    first_val = True
    file_data = []

    # Comment out for now to test the times file sync (bias = 0)

    if (lines_limit > 0):
        for i, line in enumerate(log_file):
            if (i < lines_limit):
                line_data = get_data_from_line(line)
                line_data = utils.to_polar(line_data[1:])
                file_data.append(line_data)
            else:
                break
    else:
        for line in log_file:
            line_data = get_data_from_line(line)
            if (to_polar):
                line_data[1:] = utils.to_polar(line_data[1:])
            file_data.append(line_data)

    return np.array(file_data).T


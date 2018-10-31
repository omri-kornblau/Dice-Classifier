"""Get data out of a given file, specificly the dice log file."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os


def get_data_from_line(log_line):
    """Get a string and return array of numbers in it."""
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

    time = raw_data[0]*360 + raw_data[1]*60 + raw_data[2] + raw_data[3]*0.001
    x_acc = raw_data[4]
    y_acc = raw_data[5]
    z_acc = raw_data[6]

    return (np.array([time, x_acc, y_acc, z_acc]))


def get_data_from_file(log_file, lines_limit=0):
    """Run on each line in a file and create an array of the numbers in it."""
    file_data = []
    # cheking here to get rid of the inner check in case of no limit
    if (lines_limit > 0):
        for i, line in enumerate(log_file):
            if (i < lines_limit):
                file_data.append(get_data_from_line(line))
            else:
                break
    else:
        for line in log_file:
            file_data.append(np.array(get_data_from_line(line)))

    return np.array(file_data)


def seperate_throw_data(throw_data):
    """Seperate throw data parsed by get_data using quite times."""
    pass    

log_path = "..\\..\\Dice Logger\\LoggerExe\\Log.txt"
log_file = open(log_path, 'r')

data = get_data_from_file(log_file)

# alpha = 0.9
# beta = 1 - alpha
# for index in range(len(data)-1):
#     data[index+1] = data[index]*alpha + data[index+1]*beta

data = data.T
w = savgol_filter(data[1], 101, 3)
plt.plot(data[0], data[1])
plt.plot(data[0], w)
plt.show()
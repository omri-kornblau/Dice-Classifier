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


def seperate_throw_data(throw_data, quiet_time=1.5, thres=7.5):
    """Seperate throw data parsed by get_data in the middle of quiet.

    Parameters
        throw_data: np.ndarray
            Throws accelerometer data in
            the form of [time, acc_x, acc_y, acc_z]
        quiet_time: float
            The time which defines how much quiet acc is a throw.

    """
    filtered = savgol_filter(throw_data[1], 51, 2)
    grad = np.gradient(filtered, throw_data[0])
    grad_filtered = savgol_filter(grad, 101, 2)

    output = []
    temp_data = np.array([throw_data.T[0]])
    time_passed = 0
    is_throw = False
    for index, sample in enumerate(grad_filtered[:-1]):
        temp_data = np.append(temp_data, [throw_data.T[index]], axis=0)

        if (np.abs(sample) < thres):
            time_passed += throw_data[0][index+1]-throw_data[0][index]

            if (time_passed > quiet_time):
                is_throw = True

        else:
            if (is_throw):
                is_throw = False
                output.append(temp_data)
                temp_data = np.array([temp_data[-1]])
                time_passed = 0
            time_passed = 0

    output.append(temp_data)

    return output, grad_filtered, filtered

log_path = "..\\..\\Dice Logger\\LoggerExe\\Log.txt"
log_file = open(log_path, 'r')

data = get_data_from_file(log_file)

data = data.T
abfilter = np.copy(data.T)
sep, grad_filt, filt = seperate_throw_data(data)

f, (raw_plot, sep_plot, filt_plot) = plt.subplots(3, sharex=True)

plt.grid()

for l in sep:
    sep_plot.plot(l.T[0], l.T[1])

filt_plot.plot(data[0], filt)
filt_plot.plot(data[0], grad_filt)

raw_plot.plot(data[0], data[1])
plt.show()

# alpha = 0.92
# beta = 1 - alpha
# for index in range(len(abfilter)-1):
#     abfilter[index+1] = abfilter[index]*alpha + abfilter[index+1]*beta
# abfilter = abfilter.T
# filtered = abfilter[1]

# plt.plot(data[0], filtered)
# plt.plot(data[0][1:], df)

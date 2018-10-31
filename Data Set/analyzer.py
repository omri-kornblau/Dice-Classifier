"""Get data out of a given file, specificly the dice log file."""
import numpy as np
import matplotlib.pyplot as plt
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

    return ([time, x_acc, y_acc, z_acc])


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
            file_data.append(get_data_from_line(line))

    return file_data

log_path = "..\\..\\Dice Logger\\LoggerExe\\Log.txt"
log_file = open(log_path, 'r')

print(get_data_from_file(log_file, 5))

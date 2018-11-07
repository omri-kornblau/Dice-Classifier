"""Get data out of a given file, specificly the dice log file."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os


class LogDataSet(dict):
    def __init__(self):
        pass
    # def __setitem__(self, key, item):
    #     dict.__setitem__(self, key, item)

    def get_data_from_line(self, log_line):
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

    def update_data_from_file(self, log_file, data_name, lines_limit=0):
        """Run on each line in a file and create a list of the numbers in it."""
        # cheking here to get rid of the inner check in case of no limit
        file_data = []
        if (lines_limit > 0):
            for i, line in enumerate(log_file):
                if (i < lines_limit):
                    file_data.append(self.get_data_from_line(line))
                else:
                    break
        else:
            for line in log_file:
                file_data.append(np.array(self.get_data_from_line(line)))

        self[data_name] = np.array(file_data).T

    def get_throw(self, data_name='throw', quiet_time=1.5, thres=7.5):
        """Get the data of a throw based on quiet times.

        Parameters
            data_name: string
                name of the data set, need to match one that exits
            quiet_time: float
                The time which defines how much quiet acc is a throw.
            thres: float
                Max value of derivative quite time

        """
        output = []
        filtered = savgol_filter(self[data_name][1], 51, 2)
        grad = np.gradient(filtered, self[data_name][0])
        grad_filtered = savgol_filter(grad, 101, 2)

        temp_data = np.array([self[data_name].T[0]])
        time_passed = 0
        is_throw = False
        for index, sample in enumerate(grad_filtered[:-1]):
            temp_data = np.append(temp_data, [self[data_name].T[index]], axis=0)

            if (np.abs(sample) < thres):
                time_passed += self[data_name][0][index+1]-self[data_name][0][index]

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

        return output

log_path = "..\\..\\Dice Logger\\LoggerExe\\Log.txt"
log_file = open(log_path, 'r')

data = LogDataSet()
data.update_data_from_file(log_file, 'throw')

f , (xa, ya, za) = plt.subplot(, sharex=True)

xa.plot(data['throw'][1])
ya.plot(data['throw'][2])
za.plot(data['throw'][3])
print(data.get_throw())
plt.show()

"""Get data out of a given file, specificly the dice log file."""
import numpy as np
import scipy.signal as signal


class LogDataSet(dict):
    """Get and classify logger data, A dictionray of datasets."""

    def __init__(self):
        """Initialize empty data set object."""
        pass
    # def __setitem__(self, key, item):
    #     dict.__setitem__(self, key, item)

    def get_data_from_line(self, log_line):
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

        x_acc = raw_data[4]
        y_acc = raw_data[5]
        z_acc = raw_data[6]

        return (np.array([time, x_acc, y_acc, z_acc]))

    def add_dataset_from_file(self, log_file, data_name, lines_limit=0):
        """Create ndarray object from accelerations log file.

        The data is stored inside given data set name in the object.
        """
        # cheking here to get rid of the inner check in case of no limit
        file_data = []
        if (lines_limit > 0):
            for i, line in enumerate(log_file):
                if (i < lines_limit):
                    file_data = file_data.append(self.get_data_from_line(line))
                else:
                    break
        else:
            for line in log_file:
                file_data.append(self.get_data_from_line(line))
        self[data_name] = np.array(file_data).T

    def get_throw_data(self, data_name='throw', quiet_time=0.5, thres=50, filter_config=(3, 0.15)):
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
        b, a = signal.butter(*filter_config)

        #filtered = signal.savgol_filter(self[data_name][1], 5, 3)
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, self[data_name][1], zi=zi*self[data_name][1][0])
        #z2, _ = signal.lfilter(b, a, z, z[0])
        filtered = signal.filtfilt(b, a, self[data_name][1])
        # print(self[data_name][0])
        grad = np.gradient(filtered, self[data_name][0])
        #grad = signal.savgol_filter(grad, 51, 3)
        temp_data = np.array([self[data_name].T[0]])
        time_counter = 0
        is_throw = False
        for index, sample in enumerate(grad[:-1]):
            temp_data = np.append(temp_data, [self[data_name].T[index]], axis=0)

            if (np.abs(sample) < thres):
                time_counter += self[data_name][0][index+1] - self[data_name][0][index]

                if (time_counter > quiet_time):
                    is_throw = True

            else:
                if (is_throw):
                    is_throw = False
                    output.append(temp_data)
                    temp_data = np.array([temp_data[-1]])
                    time_counter = 0
                time_counter = 0

        output.append(temp_data)

        return output, filtered, grad

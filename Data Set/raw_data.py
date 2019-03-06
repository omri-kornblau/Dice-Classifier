"""Get data out of a given file, specificly the dice log file."""
import numpy as np
import scipy.signal as signal
import utils

class LogDataSet(dict):
    """Get and classify logger data, A dictionray of datasets."""

    def __init__(self):
        """Initialize empty data set object."""
        pass
    # def __setitem__(self, key, item):
    #     dict.__setitem__(self, key, item)

    def get_data_from_line(self, log_line, bias_time=0):
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
        time = raw_data[0]*360 + raw_data[1]*60 + raw_data[2] + raw_data[3]*0.001 - bias_time

        return (np.array([time, raw_data[4], raw_data[5], raw_data[6]]))

    def add_dataset_from_file(self, log_file, data_name, lines_limit=0, to_polar=False):
        """Create ndarray object from accelerations log file.

        The data is stored inside given data set name in the object.
        """
        first_val = True
        time_bias = 0
        file_data = []

        # Comment out for now to test the times file sync (bias = 0)
        # time_bias = self.get_data_from_line(str(log_file.readline()))[0]

        if (lines_limit > 0):
            for i, line in enumerate(log_file):
                if (i < lines_limit):
                    line_data = self.get_data_from_line(line, time_bias)
                    line_data = utils.to_polar(line_data[1:])
                    file_data.append(line_data)
                else:
                    break
        else:
            for line in log_file:
                line_data = self.get_data_from_line(line, time_bias)
                if (to_polar):
                    line_data[1:] = utils.to_polar(line_data[1:])
                file_data.append(line_data)

        self[data_name] = np.array(file_data).T

    def get_throw_data(self, data_name='throw', quiet_time=0.5, thres=50, filter_config=(3, 0.15), throw_times=[]):
        """Get the data of a throw based on quiet times.

        Parameters
            data_name: string
                name of the data set, from one of the datasets that where loaded.
            quiet_time: float
                Defines how long should a quiet time be to be considered as a rest after a throw.
            thres: float
                Max value of derivative quiet time.
            filter_config: tuple
                LP filter parameter set (for the method 'butter' in scipy.signal).
            throw_times: list
                User defined times to trim throws.

        """
        # Init var to store the final output (a list of the throws from the data)
        output = []

        # Filter data to get small derivatives
        b, a = signal.butter(*filter_config)

        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, self[data_name][1], zi=zi*self[data_name][1][0])

        filtered = signal.filtfilt(b, a, self[data_name][1])

        # Get the time derivative of the filtered data
        grad = np.gradient(filtered)

        # Init empty data buffer
        temp_data = np.array([self[data_name].T[0]])
        quiet_data = np.array([self[data_name].T[0]])

        # Store the time period in which the data was quiet (small derivatives)
        time_counter = 0

        current_time = 0
        throw_time_idx = 0
        is_throw = False

        for index, sample in enumerate(grad[:-1]):
            current_time = self[data_name][0][index]

            if (throw_time_idx < len(throw_times)):
                if (current_time > throw_times[throw_time_idx]):
                    if (quiet_data.shape[0] > 0):
                        quiet_data = np.array([quiet_data[-1]])
                    else:
                        temp_data = np.array([temp_data[-1]])

                    throw_time_idx += 1


            if (np.abs(sample) < thres):
                if (quiet_data.shape[0] > 0):
                    quiet_data = np.append(quiet_data, [self[data_name].T[index]], axis=0)
                else:
                    quiet_data = np.array([self[data_name].T[index]])

                time_counter += self[data_name][0][index+1] - self[data_name][0][index]

                if (time_counter > quiet_time):
                    quiet_data = np.array([])
                    is_throw = True

            else:
                if (is_throw):
                    is_throw = False
                    output.append(temp_data)
                    temp_data = np.array([])

                if (quiet_data.shape[0] > 0):
                    temp_data = np.concatenate((temp_data, quiet_data), axis=0)
                    quiet_data = np.array([])

                if (temp_data.shape[0] > 0):
                    temp_data = np.append(temp_data, [self[data_name].T[index]], axis=0)
                else:
                    temp_data = np.array([self[data_name].T[index]])

                time_counter = 0

        output.append(temp_data)

        return output, filtered, grad

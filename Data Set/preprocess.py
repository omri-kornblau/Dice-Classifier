from scipy import signal

import matplotlib.pyplot as plt
import numpy as np
import files_handler
import utils
import os

def get_cons_data(data, num_samples=50):
    output = []
    temp_data = []

    for index, sample in enumerate(data.T):
        if (index%num_samples == 0):
            output.append(np.array(temp_data))
            temp_data = []

        temp_data.append(sample)

    return output

def get_throw_data(data, quiet_time=0.5, thres=50, filter_config=(3, 0.15), throw_times=[]):
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
    z, _ = signal.lfilter(b, a, data[1], zi=zi*data[1][0])

    filtered = signal.filtfilt(b, a, data[1])

    # Get the time derivative of the filtered data
    grad = np.gradient(filtered)

    # Init empty data buffer
    temp_data = np.array([data.T[0]])
    quiet_data = np.array([data.T[0]])

    # Store the time period in which the data was quiet (small derivatives)
    time_counter = 0

    current_time = 0
    throw_time_idx = 0
    is_throw = False

    for index, sample in enumerate(grad[:-1]):
        current_time = data[0][index]

        if (throw_time_idx < len(throw_times)):
            if (current_time > throw_times[throw_time_idx]):
                if (quiet_data.shape[0] > 0):
                    quiet_data = np.array([quiet_data[-1]])
                else:
                    temp_data = np.array([temp_data[-1]])

                throw_time_idx += 1


        if (np.abs(sample) < thres):
            if (quiet_data.shape[0] > 0):
                quiet_data = np.append(quiet_data, [data.T[index]], axis=0)
            else:
                quiet_data = np.array([data.T[index]])

            time_counter += data[0][index+1] - data[0][index]

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
                temp_data = np.append(temp_data, [data.T[index]], axis=0)
            else:
                temp_data = np.array([data.T[index]])

            time_counter = 0

    output.append(temp_data)

    return output, filtered, grad

def create_fixed_length_data(data, length, label, verbose=False):
    """ Cut each one of the samples to a fixed length.

        Parameters
            length: int
                the length of the samples
    """
    result = []
    labels = []

    for (idx, sample) in enumerate(data):
        if sample.shape[0] >= length:
            result.append(np.delete(sample[len(sample)-length:], 0, 1))
            labels.append(label)
        else:
            if verbose: print("Throw %d/%d (size: %d) dumped" % (idx, len(data)-1, sample.shape[0]))
    return result, labels

def get_min_sub_size(data):
    data_sets_sizes = []
    for sub_data in data:
        data_sets_sizes.append(len(sub_data))

    return np.min(data_sets_sizes)

def load_data_from_files(files_path, num_samples=50, graph=-1, to_polar=False):
    name_to_num = {
        'throw': 0,
        'hand': 1,
        'idle': 2,
        'fakethrow': 3,
    }
    dataset = []
    targets = []

    if (graph > -1):
        g_x = plt.subplot(411)
        g_y = plt.subplot(412, sharex=g_x)
        g_z = plt.subplot(413, sharex=g_x)

        g_data = plt.subplot(414, sharex=g_x)

    for file_name in os.listdir(files_path):
        name_split = file_name.split('_')

        # Check if the file is in format (label_type_##.txt)
        if (len(name_split) == 2):
            print(file_name)
            label_type = name_split[0]

            log_file = open(os.path.join(files_path, file_name), 'r')

            raw_data = files_handler.get_data_from_file(log_file, to_polar=to_polar)
            final_data = []

            if label_type == 'throw' or label_type == 'fakethrow':
                # Fetch the relevant times file for the throw
                times_file_name = "%s_times_%s" % (label_type, name_split[1])

                with open(os.path.join(files_path, times_file_name), 'r') as times_file:
                    throw_times = files_handler.get_times_from_file(times_file)

                    final_data, fil, dfil = get_throw_data(
                        data=raw_data,
                        quiet_time=0.5,
                        thres=3,
                        filter_config=(1, 0.6),
                        throw_times=throw_times)

                    if (graph == int(name_split[1].split('.')[0])):
                        g_data.plot(raw_data[0], raw_data[1])
                        g_data.plot(raw_data[0], dfil)
                        for i in throw_times:
                            g_x.axvline(x=i)

            elif label_type == 'idle' or label_type == 'hand':
                final_data = get_cons_data(raw_data, num_samples=num_samples)

            file_data, file_targets = create_fixed_length_data(
                final_data,
                num_samples,
                name_to_num[label_type])

            dataset += file_data
            targets += file_targets

            if (graph == int(name_split[1].split('.')[0])):
                for x in final_data:
                    g_x.plot(x.T[0], x.T[1])
                    g_x.grid(b=True, which='both', axis='both')
                    g_y.plot(x.T[0], x.T[2])
                    g_z.plot(x.T[0], x.T[3])

    if (graph > -1):
        plt.show()

    return np.array(dataset), np.array(targets)

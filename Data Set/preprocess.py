from raw_data import LogDataSet
import matplotlib.pyplot as plt
import numpy as np
import utils

def create_ndarray_from_data (data, length):
    """ Cut each one of the samples to a fixed length.

        Parameters
            length: int
                the length of the samples
    """
    result = []
    for (idx, example) in enumerate(data):
        if example.shape[0] > length:
            result.append(example[:length])
        else:
            print("Throw %d/%d (size: %d) dumped" % (idx, len(data)-1, example.shape[0]))
    return np.array(result)

def get_min_sub_size (data):
    data_sets_sizes = []
    for sub_data in data:
        data_sets_sizes.append(len(sub_data))

    return np.min(data_sets_sizes)

def load_data_from_files(log_files_paths, times_files_paths, num_samples=100, graph=-1, to_polar=False):
    raw_data = LogDataSet()

    data = []

    if (graph > -1):
        g_x = plt.subplot(411)
        g_y = plt.subplot(412, sharex=g_x)
        g_z = plt.subplot(413, sharex=g_x)

        g_data = plt.subplot(414, sharex=g_x)

    for index, log_path in enumerate(log_files_paths):
        log_file = open(log_path, 'r')

        raw_data.add_dataset_from_file(log_file, index, to_polar=to_polar)
        throw_times = utils.get_times_from_file(times_files_paths[index])

        throw_data, fil, dfil = raw_data.get_throw_data(
            data_name=index,
            thres=2,
            filter_config=(1, 0.6),
            throw_times=throw_times)

        data.append(create_ndarray_from_data(throw_data, num_samples))

        if (graph == index):
            for x in throw_data:
                g_x.plot(x.T[0], x.T[1])
                g_x.grid(b=True, which='both', axis='both')
                g_y.plot(x.T[0], x.T[2])
                g_z.plot(x.T[0], x.T[3])

    sub_size = get_min_sub_size(data)

    y = np.array([], dtype=np.int32)
    for index, sub_data in enumerate(data):
        temp_array = np.ones(sub_size, dtype=np.int32)*index
        y = np.concatenate((y, temp_array))
        sub_data = sub_data[:sub_size]

    np.random.seed(31)
    np.random.shuffle(y)

    X = []
    for index in y:
        X.append(data[index][0])
        data[index] = np.delete(data[index], 0, axis=0)

    X = np.array(X)

    if (graph > -1):
        g_data.plot(raw_data[graph][0], raw_data[graph][1])
        g_data.plot(raw_data[graph][0], dfil)
        plt.show()

    return X, y

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from raw_data import LogDataSet

log_path = "..\\..\\Dice Logger\\LoggerExe\\Log.txt"
log_file = open(log_path, 'r')
data = LogDataSet()
data.add_dataset_from_file(log_file, 'throw')

throw_data, fil, dfil = data.get_throw_data(thres=100, filter_config=(1, 0.15))

g_orig = plt.subplot(311)
g_f = plt.subplot(312, sharex=g_orig)
g_fft = plt.subplot(313)
g_fft = plt.title("FFT of each throw")

fft = []
for x in throw_data:
    fft.append(np.fft.fft(x.T[1], 200))
    g_orig.plot(x.T[0], x.T[1])
    g_fft.plot(fft[-1])

g_f.plot(data['throw'][0], dfil/10)
g_f.plot(data['throw'][0], fil)

plt.show()


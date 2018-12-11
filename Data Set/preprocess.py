from raw_data import LogDataSet
import matplotlib.pyplot as plt
import numpy as np
import numpy as np

log_path = "..\\..\\Dice Logger\\LoggerExe\\One_Soft_One_Hard_With_Breaks.txt"
log_file = open(log_path, 'r')
data = LogDataSet()

data.add_dataset_from_file(log_file, 'throw')
throw_data, fil, dfil = data.get_throw_data(
    data_name='throw',
    thres=2,
    filter_config=(1, 0.6))

g_x = plt.subplot(411)
g_y = plt.subplot(412, sharex=g_x)
g_z = plt.subplot(413, sharex=g_x)
g_data = plt.subplot(414, sharex=g_x)

bound = 50
in_range = lambda x: ((x < bound) and (x > -bound))

fft = []
y = []

g_data.plot(data['throw'][0], fil)
g_data.plot(data['throw'][0], dfil)

for x in throw_data:
    g_x.plot(x.T[0], x.T[1])
    g_y.plot(x.T[0], x.T[2])
    g_z.plot(x.T[0], x.T[3])

print("Throw amount:", len(throw_data))
# g_f.plot(data['throw'][0], dfil/10)
# g_f.plot(data['throw'][0], fil)

plt.show()


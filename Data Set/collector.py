"""
Small code for getting time stamps when collecting data
"""

import numpy as np
import time

inp = ''
start_time = time.time()

local_time = time.localtime()
log_format_start_time = local_time.tm_hour*360 \
    + local_time.tm_min*60 \
    + local_time.tm_sec \
    + (start_time - round(start_time))

with open("Output.txt", "w") as text_file:
    text_file.write("%f\n" % log_format_start_time)
    while (inp == ''):
        inp = input()
        current_time = time.time() - start_time
        text_file.write("%f\n" % current_time)


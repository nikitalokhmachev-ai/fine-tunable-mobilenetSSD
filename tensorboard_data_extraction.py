# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:49:30 2020

@author: N1kkQ
"""

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

PATH_TO_EVENT = 'training/events.out.tfevents.1585830994.DESKTOP-HDQ7S5A'

ea = event_accumulator.EventAccumulator(PATH_TO_EVENT,
    size_guidance={ # see below regarding this argument
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 1,
})

ea.Reload()

data = ea.Scalars('Losses/TotalLoss')

steps = [el[1] for el in data][1::]
values = np.array([el[2] for el in data][1::])

values_mean = values.mean()
values_std = values.std()
values_median = np.median(values)

x_1 = [steps[0], steps[-1]]
y_1 = np.array([values_median, values_median])

y_high = y_1 + values_std
y_low = y_1 - values_std

ma = []
md = []
n = 3
ma_steps = steps[n::]

for i in range(n, len(values)):
    ma_cur = 0
    md_cur = 0
    
    for j in range(0, n):
        ma_cur += values[i-j]
    ma_cur = ma_cur/n
    
    for j in range(0, n):
        md_cur += (values[i-j] - ma_cur)**2
    md_cur = (md_cur/(n-1))**0.5
    
    md.append(md_cur)
    ma.append(ma_cur)
    
ma = np.array(ma)
md = np.array(md)

md_high = ma + md
md_low = ma - md
#plt.plot(steps, values, 'r', x_1, y_1, 'k', x_1, y_high, 'c', x_1, y_low, 'c')
plt.plot(steps, values, 'r', ma_steps, ma, 'g', ma_steps, md_high, 'b', ma_steps, md_low, 'b')
plt.legend(['Функция потерь','Скользящее среднее','Скользящее СКО'])
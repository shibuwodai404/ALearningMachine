#This program is for extracting special channel's signal of EEG
#And also, Using STFT to process data and get the feature

import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import pandas as pd
import os
import mne
import re
from scipy import signal, fft
import scipy.signal as signal
from mne.preprocessing import ica
from mne.preprocessing import create_eog_epochs
from mne.io import concatenate_raws
from mne import Epochs
from mne.time_frequency import psd_welch

'''
此程序用于对不同动作的EEG数据，提取cz、c3、c4三个影响较大的电极信号提取特征，并保存特征图片，用于输入CNN网络
'''

data_path = r'/ '
save_path = '/ '
FileNames = os.listdir(data_path)
for i in FileNames:
    if re.search(r'\.csv$', i):
        fullfilename = os.path.join(data_path, i)
        df = pd.read_csv(fullfilename, encoding='utf-8', usecols=[14])  #根据不同电极，修改usecols，如下所说8，11，14分别表示Cz C3 C4
        df = df.transpose()
        data1 = df

        # data1 = pd.read_csv(data_path, usecols=[7,10,13])
        #data1 = pd.read_csv(data_path, usecols=[14])  # 7、10、13 分别表示Cz C3 C4, 对原始数据要加一列初列
        #data1 = data1.transpose()

        ch_names = ['Cz']

        ch_types = ['eeg']

        # sample rate
        sfreq = 125  # Hz

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        raw = mne.io.RawArray(data1, info, verbose=False)
        print('raw:')
        print(raw)
        print("bad channels:")
        print(raw.info['bads'])  # no 'bads' channel was output.
        print(raw.info)
        # raw.plot()
        # raw.plot(duration=4, n_channels=3, clipping=None)
        # plt.show()

        # STFT——目前采用STFT将每个sub里的每个动作挑选Cz，C3，C4电极提取特征值（此时还需要增加预处理在前面），有待完善；
        fs = 125  # 采样频率
        data0 = data1  # 一维数据
        data = np.array(data0).flatten()
        print(data.shape)

        f, t, tf = signal.stft(data, fs=fs, window='hamming', nperseg=256, noverlap=None, nfft=None,
                               detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)
        print(data.shape)
        print('tf shape:')
        print(tf.shape)

        #  fs:时间序列的采样频率,  nperseg:每个段的长度，默认为256(2^n)   noverlap:段之间重叠的点数。如果没有则noverlap=nperseg/2

        # window ： 字符串或元祖或数组，可选需要使用的窗。
        # #如果window是一个字符串或元组，则传递给它window是数组类型，直接以其为窗，其长度必须是nperseg。
        # 常用的窗函数有boxcar，triang，hamming， hann等，默认为Hann窗。

        # nfft ： int，可选。如果需要零填充FFT，则为使用FFT的长度。如果为 None，则FFT长度为nperseg。默认为无

        # detrend ： str或function或False，可选
        # 指定如何去除每个段的趋势。如果类型参数传递给False，则不进行去除趋势。默认为False。

        # return_onesided ： bool，可选
        # 如果为True，则返回实际数据的单侧频谱。如果 False返回双侧频谱。默认为 True。请注意，对于复杂数据，始终返回双侧频谱。

        # boundary ： str或None，可选
        # 指定输入信号是否在两端扩展，以及如何生成新值，以使第一个窗口段在第一个输入点上居中。
        # 这具有当所采用的窗函数从零开始时能够重建第一输入点的益处。
        # 有效选项是['even', 'odd', 'constant', 'zeros', None].
        # 默认为‘zeros’,对于补零操作[1, 2, 3, 4]变成[0, 1, 2, 3, 4, 0] 当nperseg=3.

        # 填充： bool，可选
        # 指定输入信号在末尾是否填充零以使信号精确地拟合为整数个窗口段，以便所有信号都包含在输出中。默认为True。
        # 填充发生在边界扩展之后，如果边界不是None，则填充为True，默认情况下也是如此。

        # axis ： int，可选
        # 计算STFT的轴; 默认值超过最后一个轴(即axis=-1)。
        z = np.abs(tf)
        plt.pcolormesh(np.abs(tf), vmin=0, vmax=z.mean() * 5)
        # plt.pcolormesh(f, t, np.abs(tf), vmin=0, vmax=4)
        # plt.title('STFT')
        # plt.ylabel('frequency')
        # plt.xlabel('time')
        plt.axis('off')
        plt.savefig("/Users/leogy/study/HKUStudy/Dissertation/DataDownload/MILimbEEG/data/"+i+".png", bbox_inches='tight', dpi=300)
        #plt.show()
        plt.clf()


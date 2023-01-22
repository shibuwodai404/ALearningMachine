#This programme is a practice for EEG data analysing with MNE+Python

import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import pandas as pd
import os
import mne
from mne.preprocessing import ica
from mne.preprocessing import create_eog_epochs
from mne.io import concatenate_raws
from mne import Epochs
from mne.time_frequency import psd_welch
from mne.preprocessing import ICA
from mne.datasets import sample
from mne.preprocessing import create_eog_epochs, ICA
import scipy.io
from datetime import datetime
import scipy.io as scio
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import cross_val_score


data1 = pd.read_csv('  ')
print(data1.shape)
data1 = data1.transpose()
print(data1.shape)


ch_names = ['FC5', 'F3', 'Fz', 'F4','FC6', 'FC1', 'FC2', 'Cz', 'T7', 'CP5', 'C3', 'CP1', 'CP2', 'C4', 'CP6', 'T8']


#sample rate
sfreq = 125   #Hz

info = mne.create_info(ch_names = ch_names, sfreq = sfreq)

raw = mne.io.RawArray(data1, info, verbose=False)
print('raw:')
print(raw)
print("bad channels:")
print(raw.info['bads'])  #  no 'bads' channel was output.

raw.plot()
raw.plot(duration=5, n_channels=16)

############# 2. 预处理 #####################
# 过滤高低频域，过滤防止0的干扰，同时减少数据量   7-31Hz
raw.filter(l_freq=7, h_freq=25)
# 选择通道 16channels
alllist = ['FC5', 'F3', 'Fz', 'F4','FC6', 'FC1', 'FC2', 'Cz', 'T7', 'CP5', 'C3', 'CP1', 'CP2', 'C4', 'CP6', 'T8']
goodlist = ['FC5', 'F3', 'Fz', 'F4','FC6', 'FC1', 'FC2', 'Cz', 'T7', 'CP5', 'C3', 'CP1', 'CP2', 'C4', 'CP6', 'T8']
goodlist = set(goodlist)
badlist = []

for x in alllist:
    if x not in goodlist:
        badlist.append(x)
picks = mne.pick_channels(alllist, goodlist, badlist)
raw.plot(order=picks, n_channels=16, duration=4)
for x in badlist:
    raw.info['bads'].append(x)

raw.plot()
raw.plot(duration=5, n_channels=16)

#epochs = mne.Epochs(raw, tmin=-0.2, tmax=0.5, baseline=(None,0))

#electrode information
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)
raw.interpolate_bads()  # 根据蒙太奇模板做坏导的插值计算

#plot them
raw.plot_psd(fmax=50)   #fmax can not exceed 1/2 freq which is 125Hz in this sample原始数据功率葡图
raw.set_montage(montage)
#plot electrode sensors position
raw.plot_sensors(ch_type= 'eeg')
raw.plot(duration=0.1, n_channels=16)
plt.show()
raw.plot_psd_topo()
plt.show
print(raw.info)



# set up and fit the ICA
'''
raw.load_data()
ica.plot_sources(raw, show_scrollbars=False)
ica.plot_components()
'''

ica = mne.preprocessing.ICA(n_components=16, random_state=97, max_iter='auto')
ica.fit(raw)
ica.exclude = [0, 1, 7]  # details on how we picked these are omitted here
ica.plot_properties(raw, picks=ica.exclude)
ica.plot_overlay(raw, exclude=[0], picks='eeg')
'''
#from essay
ica.decompose_raw (raw, picks=picks, decim=3)  ###### from essay

scores = ica.find_sources_raw (raw, target='EOG 061', score_func='correlation')
ica.exclude += [scores.argmax()]

ica.plot_topomap(ica.exclude)
ica.plot_sources_raw (raw, ica.exclude, start=100., stop=103.)######  from essay
'''

orig_raw = raw.copy()
raw.load_data()
ica.apply(raw)

# show some frontal channels to clearly illustrate the artifact removal
chs = ['FC5', 'F3', 'Fz', 'F4','FC6', 'FC1', 'FC2', 'Cz', 'T7', 'CP5', 'C3', 'CP1', 'CP2', 'C4', 'CP6', 'T8']
chan_idxs = [raw.ch_names.index(ch) for ch in chs]
orig_raw.plot(order=chan_idxs, start=12, duration=4)
raw.plot(order=chan_idxs, start=12, duration=5)



'''
#fenerate continue data
reject_criteria = dict(mag=4000e-15,     # 4000 fT
                       grad=4000e-13,    # 4000 fT/cm
                       eeg=150e-6,       # 150 µV
                       eog=250e-6)       # 250 µV
'''

#Epoch划分
# 30s一个epoch
epochs = mne.make_fixed_length_epochs(raw, duration=30, preload=False)
epochs.plot_image(picks=['FC5'])
#对epochs数据进行求平均获取诱发响应
evoked = epochs.average()
evoked.plot(time_unit='s')
plt.show()
'''
#连接分析
epochs.load_data().filter(l_freq=8, h_freq=12)
alpha_data = epochs.get_data()
corr_matrix = envelope_correlation(alpha_data).get_data()
print(corr_matrix.shape)
first_30 = corr_matrix[0]
last_30 = corr_matrix[-1]
corr_matrices = [first_30, last_30]
color_lims = np.percentile(np.array(corr_matrices), [5, 95])
titles = ['First 30 Seconds', 'Last 30 Seconds']

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.suptitle('Correlation Matrices from First 30 Seconds and Last 30 Seconds')
for ci, corr_matrix in enumerate(corr_matrices):
    ax = axes[ci]
    mpbl = ax.imshow(corr_matrix, clim=color_lims)
    ax.set_xlabel(titles[ci])
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.2, 0.025, 0.6])
cbar = fig.colorbar(ax.images[0], cax=cax)
cbar.set_label('Correlation Coefficient')
'''

#特征提取
def eeg_power_band(epochs):
    """脑电相对功率带特征提取
    该函数接受一个""mne.Epochs"对象，
    并基于与scikit-learn兼容的特定频带中的相对功率创建EEG特征。
    Parameters
    ----------
    epochs : Epochs
        The data.
    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # 特定频带
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=30.)
    # 归一化 PSDs,这个数组中含有0元素，所以会出现问题，正确的解决方式，从epoch中去除或者从数组中去除
    # psds = np.where(psds < 0.1, 0.1, psds)
    # sm = np.sum(psds, axis=-1, keepdims=True)
    # psds = numpy.divide(psds, sm)
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)

ans = eeg_power_band(epochs)
plt.imshow(ans)
plt.show()
print(ans)
print("shape of ans:")
print(ans.shape)

# 截取前240个数据
if ans.shape[0] > 240 :
    ans = ans[:240]
print(ans.shape) ## hxx(240, 45), qyp (239, 45)

savepath = '/Users/leogy/study/HKUStudy/Dissertation/DataDownload/MILimbEEG/Tasks/'
resultName = 'S1epoch'

if not os.path.exists(savepath):
    os.mkdir(savepath)
np.save(savepath + "/" + resultName, ans)



'''
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events=events, tmin=-0.2, tmax=0.5,
                    reject=reject_criteria, preload=True)
'''


# path to subjects' MRI files
##subjects_dir = sample_data_folder / 'subjects'
# plot the STC
##stc.plot(initial_time=0.1, hemi='split', views=['lat', 'med'],
#         subjects_dir=subjects_dir)
'''
#epoch
raw.info['bads'] += [ ]  # bads + 2 more
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True, eog=True,
                       exclude='bads')

epochs = mne.Epochs(raw, events,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))

evoked = epochs.average()

evoked.plot(time_unit='s')
plt.show()
'''
'''
picks = mne.pick_types(raw.info, meg=True, exclude='bads')
t_idx = raw.time_as_index([10., 20.])
data, times = raw[picks, t_idx[0]:t_idx[1]]
plt.plot(times, data.T)
plt.title("Sample channels")
'''

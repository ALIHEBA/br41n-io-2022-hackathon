import sys
import os
import mat73
sys.path.insert(0, os.path.abspath('..'))
import math

import numpy as np
import scipy.io as sio
from sklearn.cross_decomposition import CCA
from sklearn.metrics import confusion_matrix

from scripts import ssvep_utils as su

#data_path = os.path.abspath('../data')
all_segment_data = dict()
#all_acc = list()
window_len = 1
shift_len = 1
sample_rate = 256
duration = int(window_len*sample_rate)
flicker_freq = np.array([9, 10, 12, 15])

def get_cca_reference_signals(data_len, target_freq, sampling_rate):
    reference_signals = []
    t = np.arange(0, (data_len/(sampling_rate)), step=1.0/(sampling_rate))
    reference_signals.append(np.sin(np.pi*2*target_freq*t))
    reference_signals.append(np.cos(np.pi*2*target_freq*t))
    reference_signals.append(np.sin(np.pi*4*target_freq*t))
    reference_signals.append(np.cos(np.pi*4*target_freq*t))
    reference_signals = np.array(reference_signals)
    
    return reference_signals

def find_correlation(n_components, np_buffer, freq):
    cca = CCA(n_components)
    corr = np.zeros(n_components)
    result = np.zeros(freq.shape[0])
    for freq_idx in range(0,freq.shape[0]):
        cca.fit(np_buffer.T,np.squeeze(freq[freq_idx, :, :]).T)
        O1_a,O1_b = cca.transform(np_buffer.T, np.squeeze(freq[freq_idx, :, :]).T)
        ind_val = 0
        for ind_val in range(0,n_components):
            corr[ind_val] = np.corrcoef(O1_a[: ,ind_val], O1_b[:, ind_val])[0 ,1]
            result[freq_idx] = np.max(corr)
    
    return result

def cca_classify(segmented_data, reference_templates):
    predicted_class = []
    labels = []
    for target in range(0, segmented_data.shape[0]):
        for trial in range(0, segmented_data.shape[2]):
            for segment in range(0, segmented_data.shape[3]):
                labels.append(target)
                result = find_correlation(1, segmented_data[target, :, trial, segment, :], 
                                      reference_templates)
                predicted_class.append(np.argmax(result)+1)
    labels = np.array(labels)+1
    predicted_class = np.array(predicted_class)

    return labels, predicted_class
    
for subject in np.arange(0, 3):
    eeg1_dataset = mat73.loadmat(f'C:\data/s{subject+1}.mat')
    eeg1 = np.array(eeg1_dataset['y'], dtype='float32')
    
    eeg2_dataset = sio.loadmat(f'C:\data\classInfo_4_5.mat')
    eeg2 = np.array(eeg2_dataset['A'], dtype='float32')
    
    num_classes = eeg2.shape[0]
    num_trials = eeg2.shape[1]
    num_chan = eeg1.shape[0]
    total_trial_len = eeg1.shape[1] 
    
    filtered_data = su.get_filtered_eeg(eeg1, eeg2, 3, 40, 2, sample_rate)
    all_segment_data[f's{subject+1}'] = su.get_segmented_epochs(filtered_data, window_len, 
                                                           shift_len, sample_rate)
reference_templates = []
for fr in range(0, len(flicker_freq)):
    reference_templates.append(get_cca_reference_signals(duration, flicker_freq[fr], sample_rate))
reference_templates = np.array(reference_templates, dtype='float32')

#all_acc = np.array(all_acc)
#print(f'Overall Accuracy Across Subjects: {np.mean(all_acc)*100} %, std: {np.std(all_acc)*100} %')
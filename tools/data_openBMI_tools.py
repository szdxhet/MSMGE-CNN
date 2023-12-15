import numpy as np
from scipy.io import loadmat
import os
import torch
import resampy
import scipy.signal as ss


def get_epochs( dataPath, url=None, epochWindow=[0, 4],chans=[7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20],downsampleFactor=4):
    eventCode = [1, 2]  # start of the trial at t=0
    s = 1000
    offset = 0

    # read the mat file:
    try:
        data = loadmat(dataPath)
    except:
        print('Failed to load the data. retrying the download')
        data = None

    x = np.concatenate((data['EEG_MI_train'][0, 0]['smt'], data['EEG_MI_test'][0, 0]['smt']), axis=1).astype(np.float32)
    y = np.concatenate((data['EEG_MI_train'][0, 0]['y_dec'].squeeze(), data['EEG_MI_test'][0, 0]['y_dec'].squeeze()),axis=0).astype(int) - 1
    c = np.array([m.item() for m in data['EEG_MI_train'][0, 0]['chan'].squeeze().tolist()])
    s = data['EEG_MI_train'][0, 0]['fs'].squeeze().item()
    del data

    # extract the requested channels:
    if chans is not None:
        x = x[:, :, np.array(chans)]
        c = c[np.array(chans)]

    # down-sample if requested .
    if downsampleFactor is not None:
        xNew = np.zeros((int(x.shape[0] / downsampleFactor), x.shape[1], x.shape[2]), np.float32)
        for i in range(x.shape[2]):  # resampy.resample cant handle the 3D data.
            xNew[:, :, i] = resampy.resample(x[:, :, i], s, s / downsampleFactor, axis=0)
        x = xNew
        s = s / downsampleFactor

    # change the data dimensions to be in a format: Chan x time x trials
    x = np.transpose(x, axes=(2, 0, 1))

    return {'x': x, 'y': y, 'c': c, 's': s}




def load_openBMI_data_single_subject(filename, subject_id):

    if subject_id<10:
        train_set = get_epochs(os.path.join(filename, 'sess01_subj0' + str(subject_id) + '_EEG_MI.mat'))
        test_set = get_epochs(os.path.join(filename, 'sess02_subj0' + str(subject_id) + '_EEG_MI.mat'))
    else:
        train_set = get_epochs(os.path.join(filename, 'sess01_subj' + str(subject_id) + '_EEG_MI.mat'))
        test_set = get_epochs(os.path.join(filename, 'sess02_subj' + str(subject_id) + '_EEG_MI.mat'))
    train_X = train_set['x'].transpose(2, 0, 1)  # 200,20,1000
    train_y = train_set['y']  # 200
    test_X = test_set['x'].transpose(2, 0, 1)  # 200,20,1000
    test_y = test_set['y']  # 200



    train_X = torch.tensor(train_X, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.int64)
    test_y = torch.tensor(test_y, dtype=torch.int64)



    return train_X, train_y, test_X, test_y

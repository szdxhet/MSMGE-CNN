import os
import argparse

import numpy as np
import torch
from collections import OrderedDict

from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.datautil.signalproc import bandpass_cnt, exponential_running_standardize
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
import logging
from braindecode.mne_ext.signalproc import mne_apply
import pickle

import joblib
import scipy.signal as ss


log = logging.getLogger(__name__)
log.setLevel('INFO')
logging.basicConfig(level=logging.INFO)


def main(data_dir, output_dir,bci_dir):
    '''
    parent_output_dir = os.path.abspath(os.path.join(output_dir, os.pardir))
    assert os.path.exists(parent_output_dir), \
        "Parent directory of given output directory does not exist"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get file paths:
    train_data_paths, test_data_paths = get_paths_raw_data(data_dir)

    # Frequency filter low cut.

    # Process and save data.
    save_processed_dataset(train_data_paths, test_data_paths, output_dir)
    '''
    for i in range(1, 10):
        train_X, train_y, test_X, test_y =load_bciciv2a_data_single_subject(bci_dir,subject_id=i, to_tensor=False)
        save_pth = '../bci_data/subject' + str(i)
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)
        np.save(save_pth + "/train_X.npy", train_X)
        np.save(save_pth + "/train_y.npy", train_y)
        np.save(save_pth + "/test_X.npy", test_X)
        np.save(save_pth + "/test_y.npy", test_y)
    print("OK")



def get_paths_raw_data(data_dir):
    subject_ids = [x for x in range(1, 10)]

    train_data_paths = [{'gdf': data_dir + f"/A0{subject_id}T.gdf",
                         'mat': data_dir + f"/A0{subject_id}T.mat"}
                        for subject_id in subject_ids]
    test_data_paths = [{'gdf': data_dir + f"/A0{subject_id}E.gdf",
                        'mat': data_dir + f"/A0{subject_id}E.mat"}
                       for subject_id in subject_ids]

    return train_data_paths, test_data_paths


def save_processed_dataset(train_filenames, test_filenames, output_dir):
    train_data, test_data = {}, {}
    for train_filename, test_filename in zip(train_filenames, test_filenames):
        subject_id = train_filename['mat'].split('/')[-1][2:3]
        log.info("Processing data...")

        full_train_set = process_bbci_data(train_filename['gdf'],
                                           train_filename['mat'])
        test_set = process_bbci_data(test_filename['gdf'],
                                     test_filename['mat'])

        train_data[subject_id] = {'X': full_train_set.X, 'y': full_train_set.y}
        test_data[subject_id] = {'X': test_set.X, 'y': test_set.y}
        log.info(f"Done processing data subject {subject_id}\n")
    log.info("Saving processed data...")
    with open(os.path.join(output_dir, 'bciciv_2a_train.pkl'), 'wb') as f:
        joblib.dump(train_data, f)
    with open(os.path.join(output_dir, 'bciciv_2a_test.pkl'), 'wb') as f:
        joblib.dump(test_data, f)


def process_bbci_data(filename, labels_filename):
    ival = [-500, 4000]
    low_cut_hz = 4
    high_cut_hz = 40
    factor_new = 1e-3
    init_block_size = 1000

    loader = BCICompetition4Set2A(filename, labels_filename=labels_filename)
    cnt = loader.load()#(25,672528)

    # Preprocessing
    try:
        cnt = cnt.drop_channels(
            ['STI 014', 'EOG-left', 'EOG-central', 'EOG-right'])
    except ValueError:
        cnt = cnt.drop_channels(
            ['EOG-left', 'EOG-central', 'EOG-right'])
    assert len(cnt.ch_names) == 22#(22,672528)

    # lets convert to millvolt for numerical stability of next operations
    cnt = mne_apply(lambda a: a * 1e6, cnt)
    cnt = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz,cnt.info['sfreq'],filt_order=3,axis=1), cnt)
    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)#X:ndaray(288,22,1125) y:(288,)


    return dataset


def tensor_bci_data_single_subject(filename,subject_id):
    path=filename+'/subject'+str(subject_id)

    train_X = np.load(path+"/train_X.npy")
    test_X = np.load(path+"/test_X.npy")
    train_y = np.load(path+"/train_y.npy")
    test_y = np.load(path+"/test_y.npy")

    train_X = torch.tensor(train_X, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.int64)
    test_y = torch.tensor(test_y, dtype=torch.int64)
    return train_X, train_y, test_X, test_y


def load_bciciv2a_data_single_subject(filename, subject_id, to_tensor=True):
    subject_id = str(subject_id)
    train_path = os.path.join(filename, 'bciciv_2a_train.pkl')
    test_path = os.path.join(filename, 'bciciv_2a_test.pkl')
    with open(train_path, 'rb') as f:
        train_data = joblib.load(f)
    with open(test_path, 'rb') as f:
        test_data = joblib.load(f)
    train_X, train_y = train_data[subject_id]['X'], train_data[subject_id]['y']
    test_X, test_y = test_data[subject_id]['X'], test_data[subject_id]['y']
    if to_tensor:
        train_X = torch.tensor(train_X, dtype=torch.float32)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.int64)
        test_y = torch.tensor(test_y, dtype=torch.int64)
    return train_X, train_y, test_X, test_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Gdf to pkl")
    parser.add_argument('-data_dir', type=str,
                        default='D:\PycharmProjects\EEGGENET\EEGGENET-main\data\\bci\\raw\BCICIV_2a_gdf',
                        help='Gdf path to load')
    parser.add_argument('-output_dir', type=str,
                        default='D:\PycharmProjects\EEGGENET\EEGGENET-main\data\\bci\\raw\BCICIV_2a_pkl',
                        help='Pkl path to save')
    parser.add_argument('-bci_dir', type=str,
                        default='D:\PycharmProjects\EEGGENET-test\EEGGENET-main\data\\bci\\raw\BCICIV_2a_pkl',
                        help='Pkl path to save')
    args_ = parser.parse_args()
    main(args_.data_dir, args_.output_dir,args_.bci_dir)

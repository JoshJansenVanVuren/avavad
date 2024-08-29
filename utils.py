# Auth: Joshua Jansen van Vueren
# Date: 2024
# NOTE: Much of this code is borrowed from
# https://github.com/NickWilkinson37/voxseg in
# order to use the same scoring strategy

from typing import Tuple, Dict
import pandas as pd
import os
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

def read_data_file(path: str) -> pd.DataFrame:
    '''Function for reading standard Kaldi-style text data files (eg. wav.scp, utt2spk etc.)

    Args:
        path: The path to the data file.

    Returns:
        A pd.DataFrame containing the enteries in the data file.
    
    Example:
        Given a file 'data/utt2spk' with the following contents:
        utt0    spk0
        utt1    spk1
        utt1    spk2

        Running the function yeilds:
        >>> print(read_data_file('data/utt2spk'))
                0       1
        0    utt0    spk0
        1    utt1    spk1
        2    utt2    spk2
    
    '''

    with open(path, 'r') as f:
        return pd.DataFrame([i.split() for i in f.readlines()], dtype=str)

def process_data_dir(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''Function for processing Kaldi-style data directory containing wav.scp,
    segments (optional), and utt2spk (optional).

    Args:
        path: The path to the data directory.

    Returns:
        A tuple of pd.DataFrame in the format (wav_scp, segments, utt2spk), where
        pd.DataFrame contain data from the original files -- see docs for read_data_file().
        If a file is missing a null value is returned eg. a directory without utt2spk would
        return:
        (wav_scp, segments, None)

    Raises:
        FileNotFoundError: If wav.scp is not found.
    '''

    files = [f for f in os.listdir(path) if os.path.isfile(f'{path}/{f}')]
    try:
        wav_scp = read_data_file(f'{path}/wav.scp')
        wav_scp.columns = ['recording-id', 'extended filename']
    except FileNotFoundError:
        print('ERROR: Data directory needs to contain wav.scp file to be processed.')
        raise
    if 'segments' not in files:
        segments = None
    else:
        segments = read_data_file(f'{path}/segments')
        segments.columns = ['utterance-id', 'recording-id', 'start', 'end']
        segments[['start', 'end']] = segments[['start', 'end']].astype(float)
    if 'utt2spk' not in files:
        utt2spk = None
    else:
        utt2spk = read_data_file(f'{path}/utt2spk')
        utt2spk.columns = ['utterance-id', 'speaker-id']
    return wav_scp, segments, utt2spk

def score(wav_scp: pd.DataFrame, sys_segs: pd.DataFrame, ref_segs: pd.DataFrame, wav_segs: pd.DataFrame = None) -> Dict[str,Dict[str,int]]:
    '''Function for calculating the TP, FP, FN and TN counts from VAD segments and ground truth reference segments.

    Args:
        wav_scp: A pd.DataFrame containing information about the wavefiles that have been segmented.
        sys_segs: A pd.DataFrame containing the endpoints produced by a VAD system.
        ref_segs: A pd.DataFrame containing the ground truth reference endpoints.
        wav_segs (optional): A pd.DataFrame containing endpoints used prior to VAD segmentation. Only
        required if VAD was applied to subsets of wavefiles rather than the full files. (Default: None)

    Return:
        A dictionary of dictionaries containing TP, FP, FN and TN counts 
    '''

    ref_segs_masks = _segments_to_mask(wav_scp, ref_segs)
    sys_segs_masks = _segments_to_mask(wav_scp, sys_segs) 
    if wav_segs is not None:
        wav_segs_masks = _segments_to_mask(wav_scp, wav_segs)
    scores = {}
    for i in ref_segs_masks:
        if wav_segs is not None:
            score_array = wav_segs_masks[i] * (ref_segs_masks[i] - sys_segs_masks[i])
            num_ground_truth_p = int(np.sum(wav_segs_masks[i] * ref_segs_masks[i]))
            num_frames = int(np.sum(wav_segs_masks[i]))
        else:
            score_array = ref_segs_masks[i] - sys_segs_masks[i]
            num_ground_truth_p = int(np.sum(ref_segs_masks[i]))
            num_frames = len(ref_segs_masks[i])
        num_ground_truth_n = num_frames - num_ground_truth_p
        num_fn = (score_array == 1.0).sum()
        num_fp = (score_array == -1.0).sum()
        num_tp = num_ground_truth_p - num_fn
        num_tn = num_ground_truth_n - num_fp
        scores[i] = {'TP': num_tp, 'FP': num_fp, 'FN': num_fn, 'TN': num_tn}
    return scores


def _segments_to_mask(wav_scp: pd.DataFrame, segments: pd.DataFrame, frame_length: float = 0.01) -> Dict[str,np.ndarray]:
    '''Auxillary function used by score(). Creates a dictionary of recording-ids to np.ndarrays,
    which are boolean masks indicating the presence of segments within a recording.

    Args:
        wav_scp: A pd.DataFrame containing wav file data in the following columns:
            [recording-id, extended filename]
        segments: A pd.DataFrame containing segments file data in the following columns:
            [utterance-id, recording-id, start, end]
        frame_length (optional): The length of the frames used for scoring in seconds. (Default: 0.01)

    Returns:
        A dictionary mapping recording-ids to np.ndarrays, which are boolean masks of the frames
        which makeup segments within a recording.

    Example:
        A 0.1 second clip with a segment starting at 0.03 and ending 0.07 would yeild a mask:
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    '''

    wav_scp = wav_scp.copy()
    segments = segments.copy()
    wav_scp['duration'] = wav_scp['extended filename'].apply(lambda x: len(wavfile.read(x)[1])/wavfile.read(x)[0]).astype(float)
    wav_scp['mask'] = round(wav_scp['duration'] / frame_length).astype(int).apply(np.zeros)
    segments['frames'] = (round(segments['end'] / frame_length).astype(int) - \
                          round(segments['start'] / frame_length).astype(int)).apply(np.ones)
    temp = wav_scp.merge(segments, how='left', on='recording-id')
    for n,_ in enumerate(temp['mask']):
        if not np.isnan(temp['start'][n]):
            temp['mask'][n][round(temp['start'][n] / frame_length):round(temp['end'][n] / frame_length)] = temp['frames'][n]
    if len(wav_scp.index) > 1:
        wav_scp['mask'] = temp['mask'].drop_duplicates().reset_index(drop=True)
    else:
        wav_scp['mask'] = temp['mask']
    return wav_scp[['recording-id', 'mask']].set_index('recording-id')['mask'].to_dict()
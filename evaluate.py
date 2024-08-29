# Auth: Joshua Jansen van Vueren
# Date: 2024
# NOTE: Much of this code is borrowed from
# https://github.com/NickWilkinson37/voxseg in
# order to use the same scoring strategy

from utils import process_data_dir, score
import os
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

# set up relevant target directories
part = "dev"
target_dir = os.path.join("transformer_scores",part)
output_dir = os.path.join(target_dir,"output")
data_dir = os.path.join(target_dir,"data")
ground_truth_dir = os.path.join(target_dir,"ground_truth")
scores_dir = os.path.join(target_dir,"scores")

# set up SCP file for new output
dev_scp = []

with open("lists/dev.lst") as f:
    for line in f:
        dev_scp.append(line.rstrip())

with open(os.path.join(output_dir,"wav.scp"),"w+") as f:
    for line in dev_scp:
        f.write(f"{line} {os.path.join(output_dir,f'{line}.wav')}\n")

# refer to VOXSEG (https://github.com/NickWilkinson37/voxseg) for more info on file structure
wav_scp, wav_segs, _ = process_data_dir(data_dir)
_, ref_segs, _ = process_data_dir(ground_truth_dir)

thresholds = np.linspace(0.,0.1,200,endpoint=False)
thresholds = np.hstack([thresholds,np.linspace(0.1,0.9,100,endpoint=False)])
thresholds = np.hstack([thresholds,np.linspace(0.9,1,200)])

fpr = []
tpr = []
thres = []

# make one segments file per threshold
for threshold in tqdm(thresholds):
    
    all_segments = []

    for audio_file in sorted(os.listdir(output_dir)):
        if audio_file == "wav.scp" or audio_file == "segments":
            # skip non audio files
            continue

        with open(os.path.join(output_dir,audio_file,f"segments_{threshold}.scp")) as f_in:
            for line in f_in:
                all_segments.append(line.rstrip())

    with open(os.path.join(output_dir,"segments"),"w+") as f:
        for seg in all_segments:
            f.write(seg + "\n")
    
    # if threshold results in all speech classification - create dummy true classification for entire file
    if (len(all_segments) == 0) and (threshold <= 0.05):
        sys_segs = wav_scp.copy()
        sys_segs['utterance-id'] = sys_segs['recording-id'] + "_0000"
        sys_segs['end'] = wav_scp['extended filename'].apply(lambda x: len(wavfile.read(x)[1])/wavfile.read(x)[0]).astype(float)
        sys_segs['start'] = 0.
        sys_segs.drop(columns=['extended filename'],inplace=True)
    # if threshold results in no speech classification - create dummy no classification for entire file
    elif (len(all_segments) == 0) and (threshold >= 0.95):
        sys_segs = wav_scp.copy() 
        sys_segs['utterance-id'] = sys_segs['recording-id'] + "_0000"
        sys_segs['start'] = 0.
        sys_segs['end'] = 0.
        sys_segs.drop(columns=['extended filename'],inplace=True)
    else:
        _, sys_segs, _ = process_data_dir(output_dir)
    
    scores = score(wav_scp, sys_segs, ref_segs, wav_segs)

    TP,FP,TN,FN = 0.,0.,0.,0.

    for ep_id in scores:
        TP += scores[ep_id]['TP']
        FP += scores[ep_id]['FP']
        TN += scores[ep_id]['TN']
        FN += scores[ep_id]['FN']

    tpr.append((TP)/(TP+FN))
    fpr.append(1 - (TN)/(TN+FP))
    thres.append(threshold)

os.makedirs(scores_dir,exist_ok=True)

np.savetxt(os.path.join(scores_dir,"tpr.txt"),tpr)
np.savetxt(os.path.join(scores_dir,"fpr.txt"),fpr)
np.savetxt(os.path.join(scores_dir,"thres.txt"),thres)
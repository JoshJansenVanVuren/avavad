# Auth: Joshua Jansen van Vueren
# Date: 2024

from model import Wav2Vec2ForVAD
from transformers import Wav2Vec2FeatureExtractor
from typing import Union,Optional,List,Dict
import torch, argparse, tqdm, os, librosa
from scipy.signal import medfilt
import numpy as np
from config import config

######################
## argument parsing ##
######################

parser = argparse.ArgumentParser(
                    description="TransformerVAD Training")
parser.add_argument("--working_dir", type=str,
                    default="./",
                    help="Directory of source files")     
parser.add_argument("--path_to_audio_file", type=str,
                    default="audio/vfjywN5CN0Y.wav",
                    help="Audio file including path.")
parser.add_argument("--segments_dir", type=str,
                    default="segments/",
                    help="Path to output segments")
parser.add_argument("--model_dir", type=str,
                    default="model_ava/",
                    help="Path to directory containing trained VAD model.")
parser.add_argument("--threshold", type=float,
                    default=0.1,
                    help="Classification threshold: `Clean-speech` if prob >= `threshold` otherwise `other`. If -1 then use a series of predefined thresholds.")
parser.add_argument("--median_filter_size", type=int,
                    default=1,
                    help="Median filter of size N is applied after threshold. N=1 is equivalent to no filter.")

######################################
# collect the command line arguments #
######################################
args = parser.parse_args()

working_dir = args.working_dir
path_to_audio_file = args.path_to_audio_file
model_dir = args.model_dir
threshold = args.threshold # threshold for classifications - pred = 1 if prob >= threshold otherwise 0
median_filter_size = args.median_filter_size # after applying threshold apply median filter.
segments_dir = args.segments_dir

sr = config['sr'] # sampling rate
pool = config['pool'] # number of audio samples to average pool in one output frame
step = config['step'] # size of step between average pool
output_target_ms = config['output_target_ms'] # milliseconds between model output

# number of Wav2Vec2 frames to pool to achieve `output_target_ms`
stacked_outputs = config['stacked_outputs']

step_ms = config['step_ms'] # step between input audio classifications
sample_length_ms = config['sample_length_ms'] # size of input audio to classify

####################################################################
#               load in pretrained model                           #
####################################################################
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config['pretrain_model_name'])
model = Wav2Vec2ForVAD.from_pretrained(model_dir,num_labels=config['num_labels'])

model.eval()

# if GPU is available then use
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# move model to GPU or keep on CPU
model.to(device)

##################################################################
# generate list of segments (dicts) containing 'input_values' to #
# be fed to model and output timesteps for endpointing           #
##################################################################

def generate_segments_for_scoring(path_to_audio_file):
    # set sampling rate  so file will be resampled to target
    # sample rate if required
    audio_file, _ = librosa.load(path_to_audio_file,sr=sr)

    # extract audio in {sample_length_ms} segments with 
    # {step_ms} overlap
    audio_length = len(audio_file) / sr * 1000

    segments = []

    start = 0
    end = sample_length_ms
    
    counter = 1
    end_flag = False
    
    while end <= audio_length:
        out_feat = {"input_values":None,"output_timestamps":None}

        audio = audio_file[int(start * sr / 1000):int(end * sr / 1000)]

        out_feat["input_values"] = feature_extractor(audio,
                                    sampling_rate=sr,return_tensors='pt').input_values
            
        input_length = out_feat["input_values"].shape[-1]

        # get number of output frames for input
        output_length = torch.ceil(model._get_feat_extract_output_lengths(input_length) / stacked_outputs).int()

        # generate respective timestamps 
        out_feat["output_timestamps"] = torch.tensor([start + (sample_length_ms / output_length)*i for i in range(output_length)])

        segments.append(out_feat)

        start += step_ms
        end += step_ms
        counter += 1

        if (end > audio_length) and (start >= audio_length):
            continue
        elif end > audio_length and not end_flag:
            end_flag = True
            end = audio_length

    return segments

segments = generate_segments_for_scoring(path_to_audio_file)

segment_predictions = []
segment_timestamps = []

# calculate the number of overlapping output frames
overlap_frames = torch.ceil(model._get_feat_extract_output_lengths((sample_length_ms - step_ms) * 16) / stacked_outputs).int()

with torch.no_grad():
    print("| LOG: Computing VAD scores for input audio files.")

    for i,segment in enumerate(tqdm.tqdm(segments)):
        output_timestamps = segment['output_timestamps']

        del segment['output_timestamps']

        for k in segment:
            segment[k] = segment[k].to(device)

        pred = model(**segment)

        probs = torch.sigmoid(pred.logits.squeeze()).to('cpu').numpy()

        if i == 0:
            # for first segment
            segment_predictions.extend(probs)
            segment_timestamps.extend(output_timestamps.to('cpu').numpy())
        else:
            # for remainder of segments

            # make sure that overlap isnt more than the actual number of predictions
            if len(probs) < overlap_frames: overlap_frames = len(probs)

            # take mean of score for overlapping scores
            if overlap_frames != 0:
                segment_predictions[-overlap_frames:] = (segment_predictions[-overlap_frames:] + probs[:overlap_frames]) / 2 
        
            # add remainder of scores
            segment_predictions.extend(probs[overlap_frames:])
            segment_timestamps.extend(output_timestamps[overlap_frames:].to('cpu').numpy())
            

if threshold == -1:
    # If default then generate segments for a series of thresholds
    thresholds = np.linspace(0.,0.1,200,endpoint=False)
    thresholds = np.hstack([thresholds,np.linspace(0.1,0.9,100,endpoint=False)])
    thresholds = np.hstack([thresholds,np.linspace(0.9,1,200)])

    print("| LOG: Converting VAD scores to segments for predefined thresholds.")
else:
    thresholds = [threshold]

    print(f"| LOG: Converting VAD scores to segments for threshold: {threshold}.")

for threshold in tqdm.tqdm(thresholds):
    
    # convert predictions list to numpy array
    segment_predictions_np = np.array(segment_predictions)

    # apply threshold and cast back to float
    segment_predictions_np = (segment_predictions_np >= threshold).astype(float)

    # apply median filter
    segment_predictions_np = medfilt(segment_predictions_np,median_filter_size)

    # store group continous segments as speech and non-speech using FST
    output_speech_segments = []

    for i in range(len(segment_predictions_np)-1):
        #############################
        # this functions like a FST #
        #############################
        if ((segment_predictions_np[i+1] == 1) and (segment_predictions_np[i]) == 0):
            # start of speech segment
            start_speech_seg = segment_timestamps[i+1]
        elif ((segment_predictions_np[i]) == 1 and i == 0):
            # also start of speech segment
            start_speech_seg = segment_timestamps[i]
        elif (segment_predictions_np[i+1] == 0) and (segment_predictions_np[i]) == 0:
            # continuation of non-speech
            pass
        elif (segment_predictions_np[i+1] == 1) and (segment_predictions_np[i]) == 1:
            # continuation of speech
            pass
        elif ((segment_predictions_np[i+1] == 0) and (segment_predictions_np[i]) == 1):
            # end of speech segment
            end_speech_seg = segment_timestamps[i+1]

            output_speech_segments.append((np.round(start_speech_seg/1000,2),np.round(end_speech_seg/1000,2)))
        elif ((segment_predictions_np[i]) == 1 and (i+1) == len(segment_predictions_np)-1):
            # also end of speech segment
            end_speech_seg = segment_timestamps[i]

            output_speech_segments.append((np.round(start_speech_seg/1000,2),np.round(end_speech_seg/1000,2)))

    segments_file_name = path_to_audio_file.split("/")[-1].split(".")[0]

    if not os.path.isdir(segments_dir):
        os.makedirs(segments_dir)

    with open(os.path.join(segments_dir,f"segments_{threshold}.scp"),"w+") as f:
        idx = 0
        for start_s,end_s in output_speech_segments:
            f.write(f"{segments_file_name}_{int(start_s*1000):07d}_{int(end_s*1000):07d} {segments_file_name} {start_s} {end_s}\n")
            idx += 1
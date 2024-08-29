# Auth: Joshua Jansen van Vueren
# Date: 2024

import math

config = {
    "sr"        : 16000,
    "pool"      : 400,
    "step"      : 320,
    "output_target_ms"  : 105, # should stack 5 wav2vec2 outputs together
    # NOTE: changing 'stacked_outputs' here will require a change in average pooling
    # structure in Wav2Vec2ForVAD as well, and then retraining
    "epochs"    : 16,
    "batch_size": 2,
    "grad_accum_steps"  : 4,
    "learning_rate"     : 5e-5,
    "warmup_percent"    : 10, # warm up for 10% of training steps"
    "num_labels"        : 1,
    "step_ms"           : 20000, # step between input audio classifications
    # NOTE: if {sample_length_ms} less then {step_ms} then there is overlap - this is currently programmed
    #       to be averaged for overlapping frames but may be buggy.
    "sample_length_ms"  : 20000, # size of input audio to classify
    "pretrain_model_name" : "facebook/wav2vec2-xls-r-300m"
}

config["stacked_outputs"] = int(math.floor((config["output_target_ms"]*config["sr"]/1000 - config["pool"])/config["step"]) + 1)
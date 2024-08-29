# A Transformer-Based Voice Activity Detector

This repo contains the code to evaluate a two-class voice activity detector (VAD), which has been finetuned from the 300 million parameter variant of `Wav2Vec2-XLS-R`. Use of this VAD may be cited as follows:

```
@inproceedings{soapiesvad,
    title = {A Transformer-Based Voice Activity Detector},
    author = {Karan, B. and Jansen van V\"{u}ren, J. and Niesler, T.},
    booktitle = {Proc. Interspeech},
    year = {2024},
    address = {Kos, Greece},
}
```

## Structure

```
lists/
model_ava/
model_soapies/
results/
    transformer_results/
        dev/
            fpr.txt
            tpr.txt
            thres.txt
        test/
            fpr.txt
            tpr.txt
            thres.txt
    lstm_results/
    silero_results/
    Results.ipynb
segments/
```

 * `results/`
    * Contains the output scores for each of the evaluated models `transformer_results/`,`lstm_results/`, and `silero_results/` for both development (`dev/`) and test sets (`test/`).
    * `dev/` and `test/` contain three files contain the true positive (`tpr.txt`) and false positive rate (`fpr.txt`) at various thresholds (`thres.txt`).
    * `Results.ipynb` used for plotting ROC curves / determining performance at a specific operating point.
 * `lists/`
    * Contains the YouTube unique ID (i.e. `https://www.youtube.com/watch?v={ID}`) of the files that were used for training (`trn.lst`), development (`dev.lst`), and testing (`tst.lst`). The availability of these files, however, is not under our control and constantly changes.
 * `model_*/`
    * The weights for the transformer fine-tuned on the soapies data is stored in `model_soapies/` and in `model_ava/` for the transformer finetuned on the AVA-speech dataset.
    * In this repo the model binaries are saved via GIT LFS, [see docs for install](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).
 * `environment.yml`
    * In order to run the scripts in the repo, we've included a conda environment which can be installed by running `conda env create -f environment.yml`. Once the environment installs, it can be activated by `conda activate vad`.


## Scripts

 * `config.py`
    * Hyperparameters which are shared across scripts.
 * `evaluate.py` 
    * Used to generate false positive and true positive rates for various threshold, for ROC curve generation.
 * `model.py`
    * Contains class defining model structure (HuggingFace style).
 * `segment.py` / `run_segment.sh`
    * Runs VAD in inference mode.
 * `utils.py`
    * Additional functions used for scoring.

## Running the VAD

The `segment.py` script is used to run the VAD in inference mode, or alternatively `run_segment.sh` when segmenting multiple files. The python script can be run as shown in the code snippet below. For this example, the script will read in the audio file `audio/vfjywN5CN0Y.wav` and store the output predicted segments in `segments/vfjywN5CN0Y/segments_0.1.scp`.

```
working_dir=$(pwd)

python segment.py   --working_dir $working_dir \
                    --path_to_audio_file audio/vfjywN5CN0Y.wav \
                    --segments_dir segments/vfjywN5CN0Y/ \
                    --model_dir model_ava/ \
                    --threshold 0.1 \
                    --median_filter_size 1
```

The `segments_0.1.scp` file has the structure:

`{audio_filename}_{start_ms}_{end_ms} {audio_filename} {start s} {end s}` such as:

```
vfjywN5CN0Y_0020900_0022000 vfjywN5CN0Y 20.9 22.0
vfjywN5CN0Y_0035800_0041500 vfjywN5CN0Y 35.8 41.5
vfjywN5CN0Y_0043400_0046400 vfjywN5CN0Y 43.4 46.4
vfjywN5CN0Y_0047800_0049800 vfjywN5CN0Y 47.8 49.8
```

## Reproducing two class results on AVA speech development set

The sample above was taken from the AVA-speech development set, if run as mentioned above, the scores in the file `segments_0.1.scp` should be exactly equal to those in the folder `transformer_scores/dev/output/vfjywN5CN0Y/segments_0.1.scp`.

 1. To reproduce the AVA speech development set results, remove the current scores in `transformer_scores/dev/output/` and run:

    ```
    bash run_segment.sh
    ```

 2. Then regenerate the false positive and true positive rates across the development set (saved in `transformer_scores/dev/scores/`) by running:

    ```
    python evaluate.py
    ```

 3. These values should then result in the same AUC if copied as:
  `cp transformer_scores/dev/scores/ results/transformer_results/dev/`.
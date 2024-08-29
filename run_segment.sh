# PARAMETERS TO BE SET BY YOU - CURRENT WILL REPRODUCE AVA DEV RESULTS
working_dir=$(pwd)
model_dir=model_ava # OR model_soapies
median_filter_size=1
threshold=-1
audio_dir=transformer_scores/dev/data/

for path_to_audio_file in $(ls ${audio_dir}*.wav) ; do

    audio_file_no_ext=$(echo $path_to_audio_file | cut -d"." -f1 | rev | cut -d"/" -f1 | rev)

    segments_dir=transformer_scores/dev/output/${audio_file_no_ext}/

    python segment.py   --working_dir ${working_dir} \
                        --path_to_audio_file ${path_to_audio_file} \
                        --model_dir ${model_dir} \
                        --median_filter_size ${median_filter_size} \
                        --segments_dir ${segments_dir} \
                        --threshold ${threshold}

done

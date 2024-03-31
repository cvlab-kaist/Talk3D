# set video directory
INPUT_VID=$1 # Put each mp4 file separately in a folder
IDENTITY=$2
EG3D_PATH=models/ffhq-fixed-triplane512-128.pkl
ROOT_DIR=$3



# process audio (from mp4)
CUDA_VISIBLE_DEVICES=2 python ./preprocess_utils/data_utils/process.py $INPUT_VID $ROOT_DIR/$IDENTITY

# If you want to process any other audio dataset (OOD)
# WAVS=data/syncobama_B.wav
# python ./preprocess_utils/data_utils/process.py $WAVS --task 2 
# python ./preprocess_utils/data_utils/process.py $WAVS2 --task 2 

# crop images and segmentation
CUDA_VISIBLE_DEVICES=2 python ./preprocess_utils/vive3d_cropping.py \
--source_video $INPUT_VID \
--generator_path $EG3D_PATH \
--savepoint_path $ROOT_DIR \
--device 0

# detect wav2lip bbox
CUDA_VISIBLE_DEVICES=2 python ./preprocess_utils/detect_wav2lip_bbox.py \
--root_dir $ROOT_DIR \
--id $IDENTITY \
--source_images image



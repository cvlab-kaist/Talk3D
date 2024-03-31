INPUT_VID=data/May.mp4
IDENTITY=May
ROOT_DIR=/path/to/data/root/directory

mkdir $ROOT_DIR/$IDENTITY
sh preprocess_utils/do_audio_and_image_preprocessing.sh $INPUT_VID $IDENTITY $ROOT_DIR
sh preprocess_utils/do_au_detection.sh $INPUT_VID $IDENTITY $ROOT_DIR
sh preprocess_utils/do_vive3d_training.sh $INPUT_VID $IDENTITY $ROOT_DIR

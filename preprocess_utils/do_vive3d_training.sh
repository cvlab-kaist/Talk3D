INPUT_VID=$1 # Put each mp4 file separately in a folder
IDENTITY=$2
EG3D_PATH=models/ffhq-fixed-triplane512-128.pkl
ROOT_DIR=$3
SAVEPOINT=trial

# train vive3d
# It is recommended to choose images with different facial expressions as much as possible.
# RTX 3090 can hold only 3 frames, but recommended to use 5 frames
CUDA_VISIBLE_DEVICES=2 python personalize_generator.py --source_video $INPUT_VID \
--generator_path $EG3D_PATH \
--start_sec 0 \
--end_sec 10 \
--frame 0 \
--frame 85 \
--frame 110 \
--device 'cuda:0' \
--directory_name $SAVEPOINT

# inference vive3d
# This process takes time.
CUDA_VISIBLE_DEVICES=2 python invert_video.py --source_video $INPUT_VID --savepoint_path ./savepoints/${IDENTITY}_${SAVEPOINT} \
--source_video $INPUT_VID


\cp -f savepoints/${IDENTITY}_${SAVEPOINT}/G_tune.pkl $ROOT_DIR/$IDENTITY
\cp -f savepoints/${IDENTITY}_${SAVEPOINT}/inversion_0-0_angles.pt $ROOT_DIR/$IDENTITY/inversion_0-0_angles.pt
\cp -f savepoints/${IDENTITY}_${SAVEPOINT}/inversion_w_person.pt $ROOT_DIR/$IDENTITY

# rm -rf savepoints
# rm -rf video
VID_DIR=$1
VID_NAME=$2
LOCAL_DIR=$3
SAVE_DIR=$4

mkdir data_openface
cp $VID_DIR data_openface

docker run --rm --ipc=host -v $LOCAL_DIR/data_openface/:/opt/OpenFace/build/bin/facecam_exp benbuleong/openface-cambridge ./FeatureExtraction -f facecam_exp/$VID_NAME -outroot facecam_exp/ -of $SAVE_DIR -q

# mv $ROOT_DIR/data/$IDENTITY/data_openface/au.csv $SAVE_DIR
# rm -rf $ROOT_DIR/data/$IDENTITY/data_openface
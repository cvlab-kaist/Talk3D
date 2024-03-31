INPUT_VID=$1
IDENTITY=$2
ROOT_DIR=$3

# If you don't have docker image, download with the below script
# docker pull benbuleong/openface-cambridge

mkdir $ROOT_DIR/$IDENTITY/data_openface
cp $INPUT_VID $ROOT_DIR/$IDENTITY/data_openface

docker run --rm --ipc=host -v $ROOT_DIR/$IDENTITY/data_openface/:/opt/OpenFace/build/bin/facecam_exp benbuleong/openface-cambridge ./FeatureExtraction -f facecam_exp/$IDENTITY.mp4 -outroot facecam_exp/ -of au.csv -q

mv $ROOT_DIR/$IDENTITY/data_openface/au.csv $ROOT_DIR/$IDENTITY/
rm -rf $ROOT_DIR/$IDENTITY/data_openface
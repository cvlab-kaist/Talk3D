# evaluation bash file for metric logging while training
ID=$1
CONFIG=$2
DATADIR=$3
INPUTVID=$4
EVAL_TYPE=$5
INF_TYPE=$6

cd eval/

python compare.py \
--ID $ID \
--short_configs $CONFIG \
--data_dir $DATADIR \
--input_video $INPUTVID \
--eval_type $EVAL_TYPE \
--inf_type $INF_TYPE \

cd ..

# usage : sh do_evaluation obama tmp path/to/dataroot path/to/synth_video.mp4 all novel
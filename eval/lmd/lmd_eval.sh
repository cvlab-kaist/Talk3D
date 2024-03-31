PRED=$1
GT=$2

cd lmd
python3 lmd_eval.py \
  --pred $PRED \
  --gt  $GT \
  --fps 25 \

cd ..
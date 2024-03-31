save_file="all_scores_"$1".txt"

rm $save_file
yourfilenames=`ls ../results/$1`

for eachfile in $yourfilenames
do
   echo $eachfile >> $save_file
   python3 lmd_eval.py  --pred ../results/"$1"/"$eachfile"  --gt  ../gt/"$eachfile"  --fps 25  >> $save_file
done

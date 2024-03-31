save_file="all_scores_"$1".txt"

rm $save_file

yourfilenames=`ls ../results/$1`

echo dist conf >> $save_file

for eachfile in $yourfilenames
do
   echo $eachfile >> $save_file
   python run_pipeline.py --videofile ../results/$1/$eachfile --reference wav2lip --data_dir tmp_dir
   python calculate_scores_real_videos.py --videofile ../results/$1/$eachfile  --reference wav2lip --data_dir tmp_dir >> $save_file
   rm -r tmp_dir
done

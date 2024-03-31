save_file="all_scores_tmp.txt"

rm $save_file

echo dist conf >> $save_file

eachfile=$1

echo $eachfile >> $save_file
cd syncnet_python
python run_pipeline.py --videofile $eachfile --reference wav2lip --data_dir tmp_dir
python calculate_scores_real_videos.py --videofile $eachfile  --reference wav2lip --data_dir tmp_dir >> $save_file
rm -r tmp_dir
cd ..

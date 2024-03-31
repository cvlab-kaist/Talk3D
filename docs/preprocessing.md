To preprocess video dataset, run the bash file below:

```
sh do_preprocess.sh {path/to/video.mp4} {videoname} {path/to/save/directory}
```

For instance, you can run like below:

```
sh do_preprocess.sh ./videos/May.mp4 May ./data
```

This preprocessing takes time to find camera params by inversion.

*Warning* Some videos may not be continuous and could be composed of fragmented segments. In such cases, the VIVE3D inversion process may not accurately predict the correct camera params. We recommend to verify the data before use.

Also, the inverted frames are saved at `./video` directory. We recommend checking them to ensure that the inversion process was successful

You can process additional wav files by the following code:

```
python ./preprocess_utils/data_utils/process.py /path/to/audio.wav --task 2 
```

## Quick Start

We provide a part of demo dataset in this [Links](https://works.do/GMb3yGM). You can skip the `preprocess_utils/do_vive3d_training.sh` in `do_preprocessing.sh`.
Place these files in data directory `dataroot/name`.

We also provide demo checkpoint in this [Links](https://works.do/FMb9GSa), which is trained on May.mp4. Download this and add this checkpoint directory to `--checkpoint_dir` in the script. 

You can use `--only_do_inference` flag for quick inference.
Please see `demo.sh` for the example.
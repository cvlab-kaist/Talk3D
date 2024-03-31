### Guide for downloading pre-trained weights ###

Prepare the pre-trained weights for face-parsing

```
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O models/79999_iter.pth
```

Prepare the pre-trained weights for 3DMM

```
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/exp_info.npy?raw=true -O preprocess_utils/data_utils/face_tracking/3DMM/exp_info.npy
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/keys_info.npy?raw=true -O preprocess_utils/data_utils/face_tracking/3DMM/keys_info.npy
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/sub_mesh.obj?raw=true -O preprocess_utils/data_utils/face_tracking/3DMM/sub_mesh.obj
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/topology_info.npy?raw=true -O preprocess_utils/data_utils/face_tracking/3DMM/topology_info.npy
```


Download the pre-trained weight of [syncnetv2](https://github.com/Rudrabha/Wav2Lip?tab=readme-ov-file) and [s3fd](https://github.com/yxlijun/S3FD.pytorch) with the code below:

```
wget http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model -O eval/syncnet_python/data/syncnet_v2.model
wget https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth -O eval/syncnet_python/detectors/s3fd/weights/sfd_face.pth
wget https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth -O Wav2Lip/face_detection/detection/sfd/s3fd.pth
```

Download the pre-trained [ArcFace](https://github.com/ronghuaiyang/arcface-pytorch) and VGG16 with the code below:

```
wget https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=0 -O models/model_ir_se50.pth
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt -O models.vgg16.pt
```


Download weights for landmark distance calculation (borrowed from [SadTalker](https://github.com/OpenTalker/SadTalker))

```
wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/shape_predictor_68_face_landmarks.dat -O eval/lmd/checkpoints/shape_predictor_68_face_landmarks.dat
wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/BFM_Fitting.zip
unzip BFM_Fitting.zip
mv BFM_Fitting/* eval/lmd/third_part/face3d/BFM
rm -rf BFM_Fitting*
```



Download [BFM](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details) and prepare as below:

```
mv 01_MorphableModel.mat preprocess_utils/data_utils/face_tracking/3DMM
cd preprocess_utils/data_utils/face_tracking
python convert_BFM.py
```

Download the pre-trained weight of [GFPGAN v1.3](https://github.com/TencentARC/GFPGAN?tab=readme-ov-file) from [here](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth) and place it as below:

```
mv GFPGANv1.3.pth GFPGAN/gfpgan/weights
```

Download the pre-trained weight of [lipsync expert](https://github.com/Rudrabha/Wav2Lip?tab=readme-ov-file) from [here](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Flipsync%5Fexpert%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1) and place it as below:

```
mv lipsync_expert.pth Wav2Lip/checkpoints
```


Download the pre-trained [CurricularFace](https://github.com/HuangYG123/CurricularFace) backbone from [here](https://drive.google.com/file/d/1f4IwVa2-Bn9vWLwB-bUwm53U_MlvinAj/view?usp=sharing) 
and place it as below:

```
mv CurricularFace_Backbone.pth eval/metrics/models
```

Download the pre-trained [EG3D](https://github.com/NVlabs/eg3d). We used the fixed version of EG3D (See this [issue](https://github.com/NVlabs/eg3d/issues/67)).


```
mv ffhq-fixed-triplane512-128.pkl models/
```
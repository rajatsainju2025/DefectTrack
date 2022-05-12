cd src

#Tracking Script | Image Size Required = 1024 x 512

#The tracking and training procedure is similar to that described in the official implementation of FairMOT.
#https://github.com/ifzhang/FairMOT

#A sample video is provided

export CUDA_VISIBLE_DEVICES = 4,5,6,7

python track.py --task mot --data_cfg '<Include your video path in data_cfg .json file as described in the original implementation. E.g., [./lib/cfg/video_sample.json]> --load_model './DefectTrack_pretrain_models/best.pth' --arch 'hrnet_18' --conf_thres 0.3 --exp_name Predict

cd ..

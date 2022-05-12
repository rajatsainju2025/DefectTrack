from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import sys
import logging
import argparse
import numpy as np

import torch
import time
import datetime
import json

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True,
             save_image='', frame_rate=100, seq='', model=''):

    #Start the tracking of defect clusters
    tracker = JDETracker(opt, frame_rate=frame_rate, model=model)
    results = []
    frame_id = 0

    #Directory to save result file and image directory
    save_dir = os.path.join(save_dir, seq)
    mkdir_if_missing(save_dir)

    for path, img, img0 in dataloader:

        # run tracking
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        online_scores = []

        for t in online_targets:

            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6 #1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))

        #Visualizing and saving images
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_image:
            cv2.imwrite(osp.join(save_dir, '{:06d}.png'.format(frame_id)), online_im)

        frame_id += 1

    #Save results
    save_result = save_dir + '/results.txt'
    write_results(save_result, results, data_type)

    return frame_id


def main(opt, data_root='', result_root='', seqs=('MOT16-05',), exp_name='demo',
         save_images=True, save_videos=True, show_image=True, trainval=False, model=''):

    #Update logger info
    logger.setLevel(logging.INFO)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0

    #Go through the sequences
    for seq in seqs:

    	#The output directory is the same as result root here
        output_dir = result_root
        save_dir = result_root

        logger.info('start seq: {}'.format(seq))

        #Load the test images
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'images'), opt.img_size)
        result_filename = os.path.join(result_root, seq, '{}.txt'.format('results'))
        frame_rate = 100

        nf = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate, seq=seq,
                              save_image=save_images, model=model)
        n_frame += nf

        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%06d.png -r 10 -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)


if __name__ == '__main__':

    #Set the GPUs to run the experiment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    opt = opts().init()
    
    #Get the test path
    f = open(opt.data_cfg)
    data_config = json.load(f)
    f.close()

    # Creating a folder to save all the data along with the model
    ts = time.time()
    TIME_STAMP = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    result_root = os.path.join('./results', 'DefectTrack', opt.exp_name, TIME_STAMP)
    mkdir_if_missing(result_root)

    # Calling the main function
    data_root = data_config['root']
    valSet_paths = data_config['predict']
    seqs_raw = [valSet_paths[el].replace('.test', '') for el in valSet_paths]
    seqs = [seqElement.replace(data_root+'/', '') for seqElement in seqs_raw]

    #Run the main file
    main(opt,
         data_root=data_root,
         result_root=result_root,
         seqs=seqs,
         exp_name=opt.exp_name,
         show_image=False,
         save_images=True,
         save_videos=False,
         trainval=False,
         model='')
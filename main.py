import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from attack.video_attack import targeted_video_attack, untargeted_video_attack
from inception_i3d.pytorch_i3d import InceptionI3d
from model_wrapper.image_model_wrapper import ResNetFeatureExtractor, DensenetFeatureExtractor, \
    TentativePerturbationGenerator
from model_wrapper.vid_model_top_k import InceptionI3D_K_Model
from utility.args_parser import video_attack_args_parse


def main():
    args = video_attack_args_parse()

    # parameters setting
    untargeted = args.untargeted
    rank_transform = not args.no_rank_transform
    random_mask = args.random_mask
    sigma = args.sigma
    sample_per_draw = args.sample_per_draw
    image_split = args.image_split
    sub_num_sample = args.sub_num_sample
    gpus = args.gpus

    multiple_gpus = len(gpus) > 1
    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(gpu) for gpu in gpus])

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(message)s')

    # Model Initialization
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('inception_i3d/models/rgb_imagenet.pt'))
    i3d.cuda()
    i3d.eval()
    if multiple_gpus:
        i3d = nn.DataParallel(i3d).cuda()

    vid_model = InceptionI3D_K_Model(i3d)

    # if multi gpus, the tentative perturbation generator will not use the first gpu.
    if multiple_gpus:
        advs_device = list(range(1, len(args.gpus)))
    else:
        advs_device = [0]

    layer = ['fc']
    extractors = []
    need_preprocess = True
    if 'resnet50' in args.image_models:
        resnet50 = models.resnet50(pretrained=True)
        resnet50_extractor = ResNetFeatureExtractor(resnet50, layer).eval().cuda()
        if multiple_gpus:
            resnet50_extractor = nn.DataParallel(resnet50_extractor, advs_device).eval().cuda(advs_device[0])
        extractors.append(resnet50_extractor)

    if 'densenet121' in args.image_models:
        densenet121 = models.densenet121(pretrained=True).eval()
        densenet121_extractor = DensenetFeatureExtractor(densenet121, layer).eval().cuda()
        if multiple_gpus:
            densenet121_extractor = nn.DataParallel(densenet121_extractor, advs_device).eval().cuda(advs_device[0])
        extractors.append(densenet121_extractor)

    if 'densenet169' in args.image_models:
        densenet169 = models.densenet169(pretrained=True).eval()
        densenet169_extractor = DensenetFeatureExtractor(densenet169, layer).eval().cuda()
        if multiple_gpus:
            densenet169_extractor = nn.DataParallel(densenet169_extractor, advs_device).eval().cuda(advs_device[0])
        extractors.append(densenet169_extractor)

    directions_generator = TentativePerturbationGenerator(extractors, part_size=32, preprocess=need_preprocess,
                                                          device=advs_device[0])

    # Attack...
    vid = np.load(args.video)
    vid = torch.tensor(vid, dtype=torch.float, device='cuda')
    vid_label = args.label

    target_vid = np.load(args.target_video)
    target_vid = torch.tensor(target_vid, dtype=torch.float, device='cuda')
    target_label = args.target_label

    if not untargeted:
        directions_generator.set_targeted_params(target_vid.cuda(), random_mask)
        res, iter_num, adv_vid = targeted_video_attack(vid_model, vid, target_vid, directions_generator,
                                                       target_label, rank_transform=rank_transform,
                                                       image_split=image_split,
                                                       sub_num_sample=sub_num_sample, sigma=sigma,
                                                       eps=0.05, max_iter=300000,
                                                       sample_per_draw=sample_per_draw)
    else:
        directions_generator.set_untargeted_params(vid.cuda(), random_mask, scale=5.)
        res, iter_num, adv_vid = untargeted_video_attack(vid_model, vid, directions_generator,
                                                         vid_label, rank_transform=rank_transform,
                                                         image_split=image_split,
                                                         sub_num_sample=sub_num_sample, sigma=sigma,
                                                         eps=0.05, max_iter=300000,
                                                         sample_per_draw=sample_per_draw)
    adv_vid = adv_vid.cpu().numpy()
    if res:
        if untargeted:
            logging.info(
                '--------------------untargeted attack succeed using {} quries-----------------------'.format(iter_num))
        else:
            logging.info(
                '--------------------{} transfer to {} using {} quries-----------------------'.format(vid_label,
                                                                                                      target_label,
                                                                                                      iter_num))
    else:
        logging.info('--------------------Attack Fails-----------------------')

    np.save(args.adv_save_path, adv_vid)


if __name__ == '__main__':
    main()

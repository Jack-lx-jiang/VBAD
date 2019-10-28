import argparse


def video_attack_args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, required=True, help='The gpus to use')
    parser.add_argument('--untargeted', action='store_true')
    parser.add_argument('--video', default='videos/35.npy')
    parser.add_argument('--label', default=35, type=int)
    parser.add_argument('--target-video', default='videos/193.npy')
    parser.add_argument('--target-label', default=193, type=int)
    parser.add_argument('--adv-save-path', default='videos/35_adv_targeted.npy')
    parser.add_argument('--random_mask', default=0.9, type=float)
    parser.add_argument('--no_rank_transform', action='store_true')
    parser.add_argument('--sigma', type=float, default=1e-6)
    parser.add_argument('--sample_per_draw', type=int, default=48, help='Number of samples used for NES')
    parser.add_argument('--image_split', type=int, default=8)
    parser.add_argument('--image_models', nargs='+', type=str, default=['resnet50'])
    parser.add_argument('--sub_num_sample', type=int, default=12,
                        help='Number of samples processed each time. Adjust this number if the gpu memory is limited.'
                             'This number should be even and sample_per_draw can be divisible by it.')

    # pure nes, sub_num....
    args = parser.parse_args()

    for m in args.image_models:
        assert m in ['', 'resnet50', 'densenet121', 'densenet169'], print(
            'the image models must be selected from resnet50, densenet121, densenet169')

    return args


if __name__ == '__main__':
    args = video_attack_args_parse()
    print(args)

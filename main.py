# Author: Zichen Liu

import torch
from argparse import ArgumentParser
from utils.seed import set_random_seed
from utils.args import add_experiment_args, add_management_args, add_model_args, save_args
from train import trainer


def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    parser = ArgumentParser(description='DVP', allow_abbrev=False)
    args = parser.parse_known_args()[0]
    add_management_args(parser)
    add_experiment_args(parser)
    add_model_args(parser)
    args = parser.parse_args()
    
    if args.seed is not None:
        print("Setting random seed to {}".format(args.seed))
        set_random_seed(args.seed)
        
    args = save_args(args)   

    if args.mode == 'train':
        trainer.train(args)

   
if __name__ == '__main__':
    main()

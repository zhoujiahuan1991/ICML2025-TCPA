# Desc: Argument parser for the project

from argparse import ArgumentParser
import time
import os


# modularized arguments management
def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--mode', type=str, default="train", help='train mode or eval mode')
    parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
    parser.add_argument('--dataset', type=str, default="cifar100", help='Which dataset to perform experiments on.')
    parser.add_argument('--arch', type=str, default="ViT/B-16", help='The architecture of backbone.')
    parser.add_argument('--pretrained', type=str, default="imagenet21k", help='The pretrained weights of backbone.')
    parser.add_argument('--n_epochs', type=int, default=100, help='The training epochs.')
    parser.add_argument('--optimizer', type=str, default="AdamW", help='The optimizer.')
    parser.add_argument('--lr', type=float, default=3e-5, help='The learning rate.')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size.')
    parser.add_argument('--scheduler', type=str, default="cosine", help='The scheduler.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='The weight decay.')
    parser.add_argument('--transform', type=str, default='default', help='The transform method.')
    parser.add_argument("--resize_dim", type=int, default=256)
    parser.add_argument('--crop_size', default=224, type=int, help='Input size of images [default: 224].')
    parser.add_argument("--mixup", type=str, default="none")
    parser.add_argument("--cutmix_alpha", type=float, default=0)
    parser.add_argument("--pool_loss_w", type=float, default=0.1)
    parser.add_argument("--random_epoch", type=int, default=5)
    parser.add_argument("--random_prompt", action='store_true', default=False)
    parser.add_argument("--IPrompt", action='store_true', default=False)
    parser.add_argument("--simam", action='store_true', default=False)
    
    parser.add_argument('--dataset_perc', default=1.0, type=float, help='Dataset percentage for usage [default: 1.0].')
    parser.add_argument('--distributed', action='store_true', default=False, help='Whether to use the distributed mode [default: False].')
    parser.add_argument('--num_workers', type=int, default=4, help='Worker nums of data loading.')
    parser.add_argument('--pin_memory', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Num of GPUs to use.')
    
    
def add_model_args(parser: ArgumentParser) -> None:
    parser.add_argument('--model', type=str, default='VPT', help='Model name.')
    
    # for VP
    parser.add_argument('--vp_type', type=str, default="whole", help='The type of visual prompting.')
    parser.add_argument('--vp_mask', type=str, default="none", help='The mask type of visual prompting.')
    parser.add_argument('--vp_padding_n', type=int, default=10, help='The padding size of visual prompting.')
    # for VPT
    parser.add_argument('--p_len_vpt', type=int, default=10, help='The length of visual prompt.')
    parser.add_argument("--prompt_dropout_vpt", type=float, default=0.1)
    # for DVP
    parser.add_argument("--prompt_dropout_dvp", type=float, default=0.1)
    parser.add_argument('--ADVP', action='store_true', default=False)
    parser.add_argument('--RDVP', action='store_true', default=False)
    parser.add_argument('--TDVP', action='store_true', default=False)
    parser.add_argument('--topk_cls', type=int, default=3)
    parser.add_argument('--pool_size_cls', type=int, default=20)
    parser.add_argument('--len_prompts_cls', type=int, default=4)
    parser.add_argument('--topk_image', type=int, default=3)
    parser.add_argument('--pool_size_image', type=int, default=20)
    parser.add_argument('--len_prompts_image', type=int, default=4)
    

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--info', type=str, default="Debug", help='The information of the experiment.')
    parser.add_argument('--seed', type=int, default="3407", help='set the random seed')
    parser.add_argument('--output_path', type=str, default='./Output/Debug', help='The path to save the output.')
    parser.add_argument('--base_dir', type=str, default='/data/dataset/liuzichen/torchvision/')
    # for tester
    parser.add_argument('--model_load_path', type=str, default='default', help='The path of load model.')
    
      

def save_args(args):
    """
    Save the arguments into a txt file.
    """
    output_path = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '-' + args.info
    output_path = os.path.join(args.output_path, output_path)
    # 检查args.output_path是否存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # file_path = os.path.join(output_path, 'args.txt')
    # 将args保存为json文件
    import json
    with open(os.path.join(output_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # 修改args.output_path
    args.output_path = output_path
    return args
    
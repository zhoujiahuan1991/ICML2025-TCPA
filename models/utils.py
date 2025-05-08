from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn as nn
import torch
from timm.models import create_model
import timm




def get_backbone(args):
    
    if args.pretrained == 'imagenet1k':
        if args.arch == 'ViT/B-16':
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            backbone = vit_b_16(weights=weights)
        else: raise NotImplementedError()
    elif args.pretrained == 'imagenet22k':
        if args.arch == 'ViT/B-16':
            backbone = create_model(
                "vit_base_patch16_224.augreg_in21k",
                pretrained=False,
                num_classes=21843,
                drop_block_rate=None,
            )
            weight_path = '../~/models/imagenet-22k/vit_base_p16_224_in22k.pth'
            backbone.load_state_dict(torch.load(weight_path, weights_only=True), False)
        else: raise NotImplementedError()
    else: raise NotImplementedError()
    N_classes = Dataset_N_classes[args.dataset]
    backbone.reset_classifier(num_classes=N_classes)
    return backbone


class SimamModule(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-3):
        super(SimamModule, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


Dataset_N_classes = {'cifar100': 100, 
                     'cifar10': 10, 
                     'flower102': 102, 
                     'food101': 101, 
                     'FGVCAircraft': 100,
                     'EuroSAT': 10,
                     'OxfordIIITPet': 37,
                     'DTD': 47,
                     'dtd': 47,
                     'SVHN': 10,
                     'svhn': 10,
                     'GTSRB': 43,
                     'gtsrb': 43,
                     'stanford_cars': 196,
                     'cub': 200,
                     'nabirds': 555,
                     'stanford_dogs': 120,
                     'vtab-cifar(num_classes=100)':100,
                     'vtab-dtd': 47,
                     'vtab-flower': 102,
                     "vtab-pets": 37,
                     'vtab-svhn': 10,
                     'vtab-sun397': 397,
                     'vtab-caltech101': 102,
                     'vtab-cifar100':100,
                     'vtab-eurosat': 10,
                     'vtab-clevr(task="closest_object_distance")': 6,
                     'vtab-clevr(task="count_all")': 8,
                     'vtab-smallnorb(predicted_attribute="label_azimuth")': 18,
                     'vtab-smallnorb(predicted_attribute="label_elevation")': 9,
                     'vtab-dsprites(predicted_attribute="label_x_position",num_classes=16)': 16,
                     'vtab-dsprites(predicted_attribute="label_orientation",num_classes=16)': 16,
                     'vtab-kitti': 4,
                     'vtab-dmlab': 6,
}

# We use this file to import the model we want to use

from models.VPT import Model_VPT
from models.VFPT import Model_VFPT
from models.VPT_DVP import Model_VPT_DVP
from models.VPT_TCPA import Model_VPT_TCPA
from models.VFPT_TCPA import Model_VFPT_TCPA
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(args):
    if args.model == 'VPT':
        model = Model_VPT(args)
    elif args.model == 'VFPT':
        model = Model_VFPT(args)
    elif args.model == 'VPT_DVP':
        model = Model_VPT_DVP(args)
    elif args.model == 'VPT_TCPA':
        model = Model_VPT_TCPA(args)
    elif args.model == 'VFPT_TCPA':
        model = Model_VFPT_TCPA(args)
    else: raise NotImplementedError()
    
    return model.to(device)
import os
from argparse import ArgumentParser

import numpy as np
import torch

from FSC_Diff_Models import RegFSCNetDiff

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    # dest="modelpath", default='./L2_2_RegFSC_8_Smth_5.0_LR_0.0001/DiceVal_0.7544_Epoch_0300.pth',# LapIRN:923748,
                    help="frequency of saving models")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--fixed", type=str,
                    dest="fixed", default='../Data/image_A.nii',
                    help="fixed image")
parser.add_argument("--moving", type=str,
                    dest="moving", default='../Data/image_B.nii',
                    help="moving image")
opt = parser.parse_args()

savepath = opt.savepath
fixed_path = opt.fixed
moving_path = opt.moving
if not os.path.isdir(savepath):
    os.mkdir(savepath)

# imgshape_4 = (160 / 4, 192 / 4, 144 / 4)
# imgshape_2 = (160 / 2, 192 / 2, 144 / 2)
# imgshape = (160, 192, 144)
# range_flow = 0.4

start_channel = opt.start_channel

# model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
#                                                          range_flow=range_flow).cuda()
# model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
#                                                          range_flow=range_flow, model_lvl1=model_lvl1).cuda()
#
# model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
#                                                     range_flow=range_flow, model_lvl2=model_lvl2).cuda()

model = RegFSCNetDiff(2, 3, start_channel)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total number of parameters: ", count_parameters(model))

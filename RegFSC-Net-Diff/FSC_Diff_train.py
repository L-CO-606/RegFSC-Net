from torch.utils.tensorboard import SummaryWriter
import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch.nn.functional as F
import torch
from torchvision import transforms
from FSC_Diff_Models import *
import torch.utils.data as Data
import Datasets_OASIS, Trans
from natsort import natsorted
import csv

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=145001,
                    help="number of total iterations")
parser.add_argument("--local_ori", type=float,
                    dest="local_ori", default=1000.0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument("--magnitude", type=float,
                    dest="magnitude", default=0.001,
                    help="magnitude loss: suggested range 0.001 to 1.0")
parser.add_argument("--mask_labda", type=float,
                    dest="mask_labda", default=0.25,
                    help="mask_labda loss: suggested range 0.1 to 10")
parser.add_argument("--data_labda", type=float,
                    dest="data_labda", default=0.02,
                    help="data_labda loss: suggested range 0.1 to 10")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=5.0,
                    help="labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=290,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=2,
                    help="using l2 or not")
opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
local_ori = opt.local_ori
magnitude = opt.magnitude
n_c = opt.checkpoint
smooth = opt.smth_labda
mask_labda = opt.mask_labda
data_labda = opt.data_labda
using_l2 = opt.using_l2

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def dice(pred1, truth1):
    VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    dice_35=np.zeros(len(VOI_lbls))
    index = 0
    for k in VOI_lbls:
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        dice_35[index]=intersection / (np.sum(pred) + np.sum(truth))
        index = index + 1
    return np.mean(dice_35)

def s_ckp(state, save_dir, save_filename, max_model_num=10):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    # print(model_lists)
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def train():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dir = "DataPath"
    val_dir = "DataPath"

    train_composed = transforms.Compose([trans.RandomFlip(0), trans.NumpyType((np.float32, np.float32)), ])
    val_composed = transforms.Compose([trans.Seg_norm(), trans.NumpyType((np.float32, np.int16))])
    train_set = datasets_OASIS.OASISBrainDataset3(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    val_set = datasets_OASIS.OASISBrainInferDataset2(glob.glob(val_dir + '*.pkl'), transforms=val_composed)
    train_loader = Data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = Data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    model = RegFSCNet(2, 3, start_channel).cuda()
    if using_l2 == 1:
        l_sim = MSE().loss
    elif using_l2 == 0:
        l_sim = SAD().loss
    elif using_l2 == 2:
        l_sim = NCC()
    l_smo = smoothloss
    transform = SpatialTransform().cuda()
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = './Loss_{}_RegFSCNet_Diff_scale_{}_smo_{}_lr_{}/'.format(using_l2,start_channel,smooth,lr)
    csv_name = 'Loss_{}_RegFSCNet_Diff_Scale_{}_smo_{}_lr_{}.csv'.format(using_l2,start_channel,smooth,lr)
    f = open(csv_name, 'w')
    with f:
        fnames = ['Turns','DSC']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists('../logs/'+model_dir):
        os.makedirs('../logs/'+model_dir)
    sys.stdout = Logger('../logs/'+model_dir)
    tbw = SummaryWriter(log_dir='../logs/' + model_dir)
    opz, c_n, m_d = optimizer, csv_name, model_dir
    losal = np.zeros((3, iteration))
    s, e = 1, 0
    while s <= iteration:
        for a, b in train_loader:
            a, b = a.cuda().float(), b.cuda().float()
            c_1, c_2, c_3 = model(a, b)
            o = [c.squeeze().squeeze() for c in (c_1, c_2, c_3)]
            fts_o = [torch.fft.fftshift(torch.fft.fftn(c)) for c in o]
            p_fts = [F.pad(fts, (84, 84, 72, 72, 60, 60), "constant", 0) for fts in fts_o]
            d_mfs = [torch.real(torch.fft.ifftn(torch.fft.ifftshift(ift))) for ift in p_fts]
            f_xy = torch.cat([d_mf.unsqueeze(0).unsqueeze(0) for d_mf in d_mfs], dim=1)
            X_Y = transform(a, f_xy.permute(0, 2, 3, 4, 1))
            l1 = l_sim(b, X_Y)
            l5 = l_smo(f_xy)
            l = l1 + smooth * l5
            opz.zero_grad()
            l.backward()
            opz.step()
            losal[:, s] = np.array([l.item(), l1.item(), l5.item()])
            print("\rstep {} -> training loss {:.4f} - sim {:.4f} - smo {:.4f}".format(s, *losal[:, s]), end="")
            sys.stdout.flush()
            if s % n_c == 0:
                with torch.no_grad():
                    D_v = []
                    for d in val_loader:
                        model.eval()
                        v_1, v_2, v_3 = model(d[0].float().to(device), d[1].float().to(device))
                        v_fts = [torch.fft.fftshift(torch.fft.fftn(v)) for v in (v_1, v_2, v_3)]
                        v_dmfs = [torch.real(torch.fft.ifftn(torch.fft.ifftshift(ift))) for ift in v_fts]
                        vf_xy = torch.cat([vdmf.unsqueeze(0).unsqueeze(0) for vdmf in v_dmfs], dim=1)
                        w_xv_s = transform(d[2].float().to(device), vf_xy.permute(0, 2, 3, 4, 1), mod='nearest')
                        for b_i in range(bs):
                            d_bs = dice(w_xv_s[b_i, ...].data.cpu().numpy().copy(),
                                        d[3][b_i, ...].data.cpu().numpy().copy())
                            D_v.append(d_bs)
                    mn = 'DV_{:.4f}_E_{:04d}.pth'.format(np.mean(D_v), e)
                    with open(c_n, 'a') as f:
                        wr = csv.writer(f)
                        wr.writerow([e, np.mean(D_v)])
                    s_ckp(model.state_dict(), m_d, mn)
                    np.save(m_d + 'L.npy', losal)
            s += 1
            if s > iteration:
                break
        print("\none epoch passed")
        e += 1
        tbw.add_scalar('L/t', l.item(), e)
        tbw.add_scalar('DV/v', np.mean(D_v), e)
    np.save(m_d + '/L.npy', losal)

if __name__ == "__main__":
    train()


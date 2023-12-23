import glob
import os, Utils
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import datasets_OASIS, trans
from torchvision import transforms
from natsort import natsorted
from FSC_Diff_Models import *

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
                    dest="magnitude", default=1000.0,
                    help="magnitude loss: suggested range 0.001 to 1.0")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=5.0,
                    help="smth_labda loss: suggested range 0.1 to 10")
parser.add_argument("--data_labda", type=float,
                    dest="data_labda", default=0.02,
                    help="data_labda loss: suggested range 0.1 to 10")
parser.add_argument("--fft_labda", type=float,
                    dest="fft_labda", default=0.02,
                    help="fft_labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=290,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=2, #1,
                    help="using l2 or not")
opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
local_ori = opt.local_ori
magnitude = opt.magnitude
n_checkpoint = opt.checkpoint
smooth = opt.smth_labda
data_labda = opt.data_labda
using_l2 = opt.using_l2

def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    transform = SpatialTransform().cuda()
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
    test_dir = "DataPath"
    model_idx = -3
    model_dir = './Loss_{}_RegFSCNet_Diff_Scale{}_smo_{}_lr_{}/'.format(using_l2, start_channel, smooth, lr)

    dict = utils.process_label()
    if not os.path.exists('Results/'):
        os.makedirs('Results/')
    if os.path.exists('Results/'+model_dir[:-1]+'_infer.csv'):
        os.remove('Results/'+model_dir[:-1]+'_infer.csv')
    csv_writter(model_dir[:-1], 'Results/' + model_dir[:-1]+'_infer')
    line = ''
    for i in range(36):
        line = line + ',' + dict[i]
    csv_writter(line +','+'non_jec', 'Results/' + model_dir[:-1]+'_infer')

    
    model = RegFSCNetDiff(2, 3, start_channel).cuda()
    
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])#['state_dict']
    model.load_state_dict(best_model, strict=False)
    model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets_OASIS.OASISBrainInferDataset2(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    d1 = utils.AverageMeter()
    d2 = utils.AverageMeter()
    d3 = utils.AverageMeter()
    with torch.no_grad():
        i = 0
        for d in test_loader:
            model.eval()
            d = [t.cuda() for t in d]
            x, y, x_s, y_s = d[:4]
            v1, v2, v3 = model(x.float().to(device), y.float().to(device))
            v_iffts = [torch.fft.fftshift(torch.fft.fftn(v.squeeze().squeeze())) for v in (v1, v2, v3)]
            p_iffts = [F.pad(ifft, (84, 84, 72, 72, 60, 60), "constant", 0) for ifft in v_iffts]
            v_d_mfs = [torch.real(torch.fft.ifftn(torch.fft.ifftshift(ifft))) for ifft in p_iffts]
            v_f_xy = torch.cat([v_d_mf.unsqueeze(0).unsqueeze(0) for v_d_mf in v_d_mfs], dim=1)
            vf_xy = torch.cat([vdisp_mf_1.unsqueeze(0).unsqueeze(0), vdisp_mf_2.unsqueeze(0).unsqueeze(0), vdisp_mf_3.unsqueeze(0).unsqueeze(0)], dim = 1)

            D_f_xy = diff_transform(vf_xy)
            x_s_oh = nn.functional.one_hot(x_s.long(), num_classes=36).squeeze(1).permute(0, 4, 1, 2, 3).contiguous()
            d_out = transform(x_s_oh.float(), D_f_xy.permute(0, 2, 3, 4, 1))
            d_out = torch.argmax(d_out, dim=1, keepdim=True)
            
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            D_f_xy = D_f_xy.detach().cpu().numpy()
            D_f_xy[:, 0, :, :, :] = D_f_xy[:, 0, :, :, :] * D_f_xy.shape[-3] / 2
            D_f_xy[:, 1, :, :, :] = D_f_xy[:, 1, :, :, :] * D_f_xy.shape[-2] / 2
            D_f_xy[:, 2, :, :, :] = D_f_xy[:, 2, :, :, :] * D_f_xy.shape[-1] / 2

            jd = utils.jacobian_determinant_vxm(D_f_xy[0, :, :, :, :])
            l1 = utils.dice_val_substruct(d_out.long(), y_s.long(), i)
            l1 = l1 + ',' + str(np.sum(jd <= 0) / np.prod(tar.shape))
            csv_writter(l1, 'Results/' + model_dir[:-1] + '_infer')
            d3.update(np.sum(jd <= 0) / np.prod(tar.shape), x.size(0))
            print('det < 0: {:.4f}'.format(d3.avg))

            dsc_t = utils.dice_val(d_out.long(), y_s.long(), 36)
            dsc_r = utils.dice_val(x_s.long(), y_s.long(), 36)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_t.item(), dsc_r.item()))
            d1.update(dsc_t.item(), x.size(0))
            d2.update(dsc_r.item(), x.size(0))
            i += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(d1.avg, d1.std, d2.avg, d2.std))
        print('deformed det: {:.4f}, std: {:.4f}'.format(d3.avg, d3.std))


def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    main()

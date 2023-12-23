import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch

class GroupBatchnorm3d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 8,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm3d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, D, H, W = x.size()  # 3D
        x = x.view(N, self.group_num, -1, D, H, W)  # (N, group_num, -1, D, H, W)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias


class SRM(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 8,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm3d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1, 1)

        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reorganize(x_1, x_2)
        return x

    def reorganize(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRM(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv3d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv3d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # high
        self.GWC = nn.Conv3d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv3d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv3d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Separation
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Refinement
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Consolidation
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class SRCR(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRM = SRM(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRM = CRM(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRM(x)
        x = self.CRM(x)
        return x

class RegFSCNetDiff(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel): # 2, 3, 8
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(RegFSCNetDiff, self).__init__()
        self.eninput = self.Encoder(self.in_channel, self.start_channel, bias=bias_opt) # 2, 8
        self.ec1 = self.Encoder(self.start_channel, self.start_channel, bias=bias_opt)# 8, 8
        self.ec2 = self.Encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)# 8, 16
        
        self.ec3 = self.Encoder_SRCR(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)# 16 16
        
        self.ec4 = self.Encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt) # 16 32
        
        self.ec5 = self.Encoder_SRCR(self.start_channel * 4, self.start_channel * 4, bias=bias_opt) # 32 32
        
        self.ec6 = self.Encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt) # 32 64
        
        self.ec7 = self.Encoder_SRCR(self.start_channel * 8, self.start_channel * 8, bias=bias_opt) # 64 64
        
        self.ec8 = self.Encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt) # 64 128
        self.ec9 = self.Encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt) # 128 64
        self.r_dc1 = self.Encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3, stride=1, bias=bias_opt) # 128 64
        self.r_dc2 = self.Encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt) # 64 32
        self.r_dc3 = self.Encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt) # 64 32
        self.r_dc4 = self.Encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt) # 32 16
        
        self.r_dc5 = self.Encoder_SRCR(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt) # 32 32
        
        self.r_dc6 = self.Encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt) # 32 16
        self.r_dc7 = self.Encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt) # 24 16
        
        self.r_dc8 = self.Encoder_SRCR(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt) #16 16
        
        self.rr_dc9 = self.Outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False) # Outputs:  in_ch= 16 out_ch 3
        # self.r_dc10 = self.Outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.r_up1 = self.Decoder(self.start_channel * 8, self.start_channel * 8) # in_ch= 64 out_ch 64
        self.r_up2 = self.Decoder(self.start_channel * 4, self.start_channel * 4) # in_ch= 32 out_ch 32
        self.r_up3 = self.Decoder(self.start_channel * 2, self.start_channel * 2) # in_ch= 16 out_ch 16
        self.r_up4 = self.Decoder(self.start_channel * 2, self.start_channel * 2) # in_ch= 16 out_ch 16


    def Encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
            #print("ec: ", "in_ch=", in_channels, "out_ch", out_channels)
        return layer

    def Encoder_SRCR(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                SRCR(in_channels),
                # nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
            #print("ec: ", "in_ch=", in_channels, "out_ch", out_channels)
        return layer


    def Decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            # nn.Dropout(0.1),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        #print("dc: ", "in_ch=", in_channels, "out_ch", out_channels)
        return layer

    def Outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))#,
                # nn.Softsign())
            #print("Outputs: ", "in_ch=", in_channels, "out_ch", out_channels)
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        #print("e0 = self.eninput(x_in)",e0.shape) # e0 = self.eninput(x_in) torch.Size([1, 8, 160, 192, 224])
        e0 = self.ec1(e0)
        #print("e0 = self.ec1(e0) ", e0.shape) # e0 = self.ec1(e0)  torch.Size([1, 8, 160, 192, 224])

        e1 = self.ec2(e0)
        #print("e1 = self.ec2(e0) ", e1.shape) # e1 = self.ec2(e0)  torch.Size([1, 16, 80, 96, 112])
        e1 = self.ec3(e1)
        #print("e1 = self.ec3(e0) ", e1.shape) # e1 = self.ec3(e0)  torch.Size([1, 16, 80, 96, 112])

        e2 = self.ec4(e1)
        #print("e2 = self.ec4(e1) ", e2.shape) # e2 = self.ec4(e1)  torch.Size([1, 32, 40, 48, 56])
        e2 = self.ec5(e2)
        #print("e2 = self.ec5(e2) ", e2.shape) # e2 = self.ec5(e2)  torch.Size([1, 32, 40, 48, 56])

        e3 = self.ec6(e2)
        #print("e3 = self.ec6(e2) ", e3.shape) # e3 = self.ec6(e2)  torch.Size([1, 64, 20, 24, 28])
        e3 = self.ec7(e3)
        #print("e3 = self.ec7(e3) ", e3.shape) # e3 = self.ec7(e3)  torch.Size([1, 64, 20, 24, 28])

        e4 = self.ec8(e3)
        #print("e4 = self.ec8(e3) ", e4.shape) # e4 = self.ec8(e3)  torch.Size([1, 128, 10, 12, 14])
        e4 = self.ec9(e4)
        #print("e4 = self.ec9(e4) ", e4.shape) # e4 = self.ec9(e4)  torch.Size([1, 64, 10, 12, 14])

        r_d0 = torch.cat((self.r_up1(e4), e3), 1)
        #print("r_d0 = torch.cat((self.r_up1(e4), e3), 1) ", r_d0.shape)

        r_d0 = self.r_dc1(r_d0)
        #print("r_d0 = self.r_dc1(r_d0) ", r_d0.shape)
        r_d0 = self.r_dc2(r_d0)
        #print("r_d0 = self.r_dc2(r_d0) ", r_d0.shape)

        r_d1 = torch.cat((self.r_up2(r_d0), e2), 1)
        #print("r_d1 = torch.cat((self.r_up2(r_d0), e2), 1) ", r_d1.shape)

        r_d1 = self.r_dc3(r_d1)
        #print("r_d1 = self.r_dc3(r_d1) ", r_d1.shape)
        r_d1 = self.r_dc4(r_d1)
        #print("r_d1 = self.r_dc4(r_d1) ", r_d1.shape)

        # print('r_d2.shape   ', r_d2.shape)
        f_r = self.rr_dc9(r_d1)
        #print("f_r = self.rr_dc9(r_d1) ", f_r.shape)
        
        return f_r[:,0:1,:,:,:], f_r[:,1:2,:,:,:], f_r[:,2:3,:,:,:]
        # return torch.complex(f_r[:,0:1,:,:,:], f_i[:,0:1,:,:,:]), torch.complex(f_r[:,1:2,:,:,:], f_i[:,1:2,:,:,:]),torch.complex(f_r[:,2:3,:,:,:], f_i[:,2:3,:,:,:])

        # return f_xy#, f_yx

class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, flow):
        # print(flow.shape)
        d2, h2, w2 = flow.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid(
            [torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        grid_d = grid_d.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2 ** self.time_step)

        for i in range(self.time_step):
            flow_d = flow[:, 0, :, :, :]
            flow_h = flow[:, 1, :, :, :]
            flow_w = flow[:, 2, :, :, :]
            disp_d = (grid_d + flow_d).squeeze(1)
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)

            deformation = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
            flow = flow + torch.nn.functional.grid_sample(flow, deformation, mode='bilinear', padding_mode="border",
                                                          align_corners=True)
        return flow

class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        d2, h2, w2 = mov_image.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        grid_d = grid_d.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_d = flow[:,:,:,:,0]
        flow_h = flow[:,:,:,:,1]
        flow_w = flow[:,:,:,:,2]
        #Softsign
        #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
        #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
        #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
        
        #Remove Channel Dimension
        disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)

        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True)
        
        return warped

def smoothloss(y_pred):
    #print('smoothloss y_pred.shape    ',y_pred.shape)
    #[N,3,D,H,W]
    d2, h2, w2 = y_pred.shape[-3:]
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :]) / 2 * d2
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0


"""
Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""
class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=9, eps=1e-5):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class SAD:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

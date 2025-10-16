
from HyperQKAN3 import HyperQKAN

import torch 
from torch.autograd import Variable
import torch.nn.functional as F
 


def getmodel(strss, batchsize, number_class, number_point, normal):
     
    return HyperQKAN(num_classes=number_class, points=number_point,).cuda() 
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

class get_loss2(torch.nn.Module):
    def __init__(self):
        super(get_loss2, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss
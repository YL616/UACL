import torch
from modules import transform, resnet, network, contrastive_loss,my_evidence_loss,my_sdm_loss,computeLabel
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

def relu_evidence(y):
    return F.relu(y)
def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))
def softplus_evidence(y):
    return F.softplus(y)

def get_alpha(output):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    return alpha

def kl_divergence(alpha, num_classes):
    ones = torch.ones([1, num_classes], dtype=torch.float32).cuda()
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def loglikelihood_loss(y, alpha):
    y = y.cuda()
    alpha = alpha.cuda()
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step):
    y = y.cuda()
    alpha = alpha.cuda()
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32)),
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return A + kl_div

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step):
    y = y.cuda()
    alpha = alpha.cuda()  #alpha torch.Size([256, 15])
    loglikelihood = loglikelihood_loss(y, alpha)  #loglikelihood torch.Size([256, 1])
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)  #kl_div torch.Size([256, 1])
    return loglikelihood + kl_div

def e_h_coef(class_num,alpha,batchsize,epoch,total_epoch, amplitude=0.7):
    alpha = alpha.cuda()  # todo U
    u = class_num/alpha   # u torch.Size([256, 15])
    u = torch.sum(u, dim=1,keepdim=True)  #
    # print("u", u.shape)
    idx = torch.arange(batchsize)  #
    theta = 2 * (idx + 0.5) / batchsize - 1
    delta = - 2 * epoch / total_epoch + 1  #
    curve = amplitude * torch.tanh(theta * delta) + 1  #
    curve = curve.cuda()
    _, Uct_indexs = torch.sort(u, dim=1)
    Uct_indexs = Uct_indexs.cuda()
    Sorted_Curve = torch.zeros(batchsize).cuda()
    for i in range(batchsize):
        t = Uct_indexs[i]
        Sorted_Curve[i] = curve[t];
    # print("Sort", Sorted_Curve.shape, list(Sorted_Curve))
    Sorted_Curve.reshape(batchsize,-1)
    return Sorted_Curve

def use_edl_mse_loss(output, target, epoch_num, num_classes, annealing_step,total_epoch=20):
    evidence = relu_evidence(output)
    alpha = evidence + 1  #alpha torch.Size([256, 15])
    # batchsize = alpha.shape[0]
    # eh_coef = e_h_coef(num_classes,alpha,batchsize,epoch_num,total_epoch)
    # loss = torch.mean(eh_coef.mul(mse_loss(target, alpha, total_epoch, num_classes, annealing_step)))  #mse return ([256, 1])
    loss = torch.mean(mse_loss(target, alpha, total_epoch, num_classes, annealing_step))  #mse return ([256, 1])
    return loss

def use_edl_log_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step))
    return loss

def use_edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step))
    return loss


def my_one_hot_edl(c_i,c_j, epoch_num, num_classes,batchsize,total_epoch=20):  # c_i是softmax的输出结果。
    N = 2*batchsize
    c = torch.cat((c_i,c_j),dim=0)
    max_indices = torch.argmax(c, axis=1)
    one_hot_matrix = torch.zeros((N, num_classes))
    one_hot_matrix[torch.arange(N), max_indices] = 1
    output = c
    target = one_hot_matrix
    # mean_mse_loss = use_edl_mse_loss(output,target,epoch_num,num_classes,10)
    mean_edl_loss_log = use_edl_log_loss(output,target,epoch_num,num_classes,10)
    # mean_edl_loss_digamma = use_edl_digamma_loss(output,target,epoch_num,num_classes,10)
    # return mean_mse_loss
    return mean_edl_loss_log
    # return mean_edl_loss_digamma








def C3_loss(z_i, z_j, batch_size, zeta):
    z = torch.cat((z_i, z_j), dim=0)
    multiply = torch.matmul(z, z.T)   #
    # print("multiply.is_leaf", multiply.is_leaf)  # F
    a = torch.ones([batch_size])
    mask = 2 * (torch.diag(a, -batch_size) + torch.diag(a, batch_size) + torch.eye(2 * batch_size))
    mask = mask.cuda()
    # print("mask.is_leaf", mask.is_leaf)  # T
    exp_mul = torch.exp(multiply)
    numerator = torch.sum(torch.where((multiply + mask) > zeta, exp_mul, torch.zeros(multiply.shape).cuda()), dim=1)
    # u = torch.where((multiply + mask) > zeta, exp_mul, torch.zeros(multiply.shape).cuda())
    # print("uuu",list(u))
    den = torch.sum(exp_mul, dim=1)
    # print("den.is_leaf",den.is_leaf)  # F
    loss_C3 = -torch.sum(torch.log(torch.div(numerator, den))) / batch_size

    return  loss_C3



def CC_Cluster_loss(class_num, c_i, c_j):
    N = 2 * class_num
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(class_num):
        mask[i, class_num + i] = 0
        mask[class_num + i, i] = 0
    mask = mask.bool()
    # print("c_i.shape",c_i)   #torch.Size([64, 15])
    p_i = c_i.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
    p_j = c_j.sum(0).view(-1)
    p_j /= p_j.sum()
    ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
    ne_loss = ne_i + ne_j
    c_i = c_i.t()
    c_j = c_j.t()
    N = 2 * class_num
    c = torch.cat((c_i, c_j), dim=0)
    # print("c.shape", c.shape)  #torch.Size([30, 64])

    similarity_f = nn.CosineSimilarity(dim=2)
    temperature = 1.0
    one = 2*torch.ones(N)
    # weight=torch.cat((one,torch.zeros((N,28))),dim=1).cuda();
    # criterion = nn.CrossEntropyLoss(weight=weight,reduction="sum")
    criterion = nn.CrossEntropyLoss(reduction="sum")  #

    # sim.shape torch.Size([30, 30])
    sim = similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / temperature
    # print("sim.shape", sim.shape)
    sim_i_j = torch.diag(sim, class_num)
    sim_j_i = torch.diag(sim, -class_num)   # torch.Size([15])
    # print("sim_i_j.shape", sim_i_j.shape)
    positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  #torch.Size([30, 1])
    # print("positive_clusters.shape", positive_clusters.shape)
    negative_clusters = sim[mask].reshape(N, -1)     #
    # print("negative_clusters.shape", negative_clusters.shape)

    label1 = 2*torch.ones(N,1)
    label2 = torch.zeros(N,28)
    labels = torch.cat((label1,label2),dim=1).to(positive_clusters.device).float()
    # labels = torch.zeros(N).to(positive_clusters.device).long()
    logits = torch.cat((positive_clusters, negative_clusters), dim=1)
    # print("logits",logits)  # （30,29）
    # print("labels.shape",labels.shape)
    loss = criterion(logits, labels)
    loss /= N

    loss_cc_clu = loss + ne_loss
    # print("loss_cc_clu",loss_cc_clu)  # loss_cc_clu tensor(2.5754, device='cuda:0', grad_fn=<AddBackward0>)

    return loss_cc_clu
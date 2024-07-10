# -*-coding:utf-8-*-import torch
import numpy as np
import torch
import torch.nn.functional as F
from modules import transform, resnet, network, contrastive_loss,my_evidence_loss,my_sdm_loss,computeLabel,DELU_loss


def my_ECCV_loss(z_i, z_j, batch_size,num_classs,func,epoch, max_epochs, tau=0.1):
    z = torch.cat((z_i, z_j), dim=0)
    tau = torch.tensor(tau);
    evidences = torch.exp(torch.tanh(cosine_similarity_matrix) / tau)
    # print("evidences", evidences.shape, evidences)
    # print("evidences.is_leaf", evidences.is_leaf)  # T
    # sum_e = evidences + evidences.t()
    alpha_i2t = (evidences + 1).cuda()  #
    # alpha_t2i = (evidences.t() + 1).cuda()
    # print("alpha_i2t.is_leaf", alpha_i2t.is_leaf)  # T
    # alpha_i2t = (alpha_i2t+alpha_t2i)/2
    L = torch.sum(alpha_i2t, dim=1, keepdim=True)
    u = num_classs/L
    # print("u.is_leaf", u.is_leaf)  # T

    # label = computeLabel.deleteHigh_Uncertainty(cosine_similarity_matrix, u, 2*batch_size, 12, 20,10)  # label torch.Size([batchsize, baychsize])
    # # print("label.is_leaf", label.is_leaf)  # T
    # original_label = label.cuda()
    # label_num = torch.sum(original_label, dim=1, keepdim=True)
    # # print("label_num.is_leaf", label_num.is_leaf)     # T
    # h = (1 - u.detach()) * label_num
    # # h = (1 - u) * label_num
    # # print("h.is_leaf", h.is_leaf)  # T
    # S = torch.sum(alpha_i2t, dim=1, keepdim=True)
    # loss_clu = (h * (func(S) - func(alpha_i2t))).mean()
    # loss_clu = torch.div(1.0,loss_clu)
    # # print("loss_clu.is_leaf", loss_clu.is_leaf)   # T
    #
    # beta = torch.ones((1, 2* batch_size)).cuda()  # c:batchsize
    # S_alpha = torch.sum(alpha_i2t, dim=1, keepdim=True)
    # S_beta = torch.sum(beta, dim=1, keepdim=True)
    # lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha_i2t), dim=1, keepdim=True)
    # lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    # dg0 = torch.digamma(S_alpha)
    # dg1 = torch.digamma(alpha_i2t)
    # loss_kl = torch.sum((alpha_i2t - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    # loss_kl = torch.div(1.0,loss_kl.mean())
    #
    # loglikelihood_matrix = my_evidence_loss.loglikelihood_loss(cosine_similarity_matrix, original_label)
    # cross_loss = my_evidence_loss.cross_loss(cosine_similarity_matrix,original_label)
    # e_h_loss = torch.div(1,my_evidence_loss.course_function_loss(epoch, max_epochs, 2*batch_size, u, loglikelihood_matrix))
    # print("e_h:",e_h_loss)


    exp_mul = torch.exp(cosine_similarity_matrix)

    # u_matrix = num_classs/alpha_i2t   #u_matrix torch.Size([64, 64])
    u_matrix = 2/alpha_i2t
    # u_factor = torch.where(u_matrix<0.05,1-u_matrix,torch.zeros(exp_mul.shape).cuda())  #u_factor torch.Size([64, 64])
    u_factor = torch.where(u_matrix<0.01,1-u_matrix,torch.zeros(exp_mul.shape).cuda())
    # c = u_factor.mul(exp_mul)
    # print("ccccc", list(c))
    # print("1-u",(1-u_matrix).shape,(1-u_matrix))
    numerator = torch.sum(u_factor.mul(exp_mul), dim=1)
    den = torch.sum(exp_mul, dim=1)
    # den = torch.sum(torch.where(exp_mul>0.4,exp_mul,torch.zeros(exp_mul.shape).cuda()),dim=1)  #
    loss_my_C3 = -torch.sum(torch.log(torch.div(numerator, den))) / batch_size

    #


    return loss_my_C3

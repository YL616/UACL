# -*-coding:utf-8-*-
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


def compute_cosine_similarity(z_i,z_j,batch_size):
    cosine_similarity_matrix = np.zeros((batch_size,batch_size))
    for i in range(batch_size):
        for j in range(batch_size):
            cosine_similarity_matrix[i][j] = torch.cosine_similarity(z_i[i], z_j[j], dim=0, eps=1e-08)  
    # print("cosine_similarity_matrix:", cosine_similarity_matrix);
    cosine_similarity_matrix = torch.tensor(cosine_similarity_matrix)
    return cosine_similarity_matrix

def new_compute_cosine_similarity_2batchsize(z_i,z_j):
    z = torch.cat((z_i, z_j), dim=0)  
    multiply = torch.matmul(z, z.T)  
    cosine_similarity_matrix = torch.tensor(multiply)
    return cosine_similarity_matrix


def my_get_alpha(similarity_matrix, tau):
    similarity_matrix = torch.tensor(similarity_matrix);
    tau = torch.tensor(tau);
    evidences = torch.exp(torch.tanh(similarity_matrix) / tau)  
    
    #print("evidences:",evidences);
    sum_e = evidences + evidences.t()
    norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)  
    alpha_i2t = evidences + 1 
    alpha_t2i = evidences.t() + 1
    # 打印
    # print("alpha_i2t的形状:", alpha_i2t.shape);   
    # print("alpha_i2t:", alpha_i2t); 
    #print("alpha_t2i:", alpha_t2i);
    # print("ecidnce：", evidences.shape, list(evidences))
    # print("alpha：", alpha_i2t.shape, list(alpha_i2t))
    sims_tanh = torch.tanh(similarity_matrix)   
    return alpha_i2t, alpha_t2i, norm_e, sims_tanh #, similarity_matrix

def my_get_alpha_relu(similarity_matrix):
    similarity_matrix = torch.tensor(similarity_matrix);
    evidences = F.relu(similarity_matrix)
    sum_e = evidences + evidences.t()
    norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)  
    alpha_i2t = evidences + 1
    alpha_t2i = evidences.t() + 1
    # sims_tanh = torch.tanh(similarity_matrix)
    return alpha_i2t, alpha_t2i, norm_e  #, sims_tanh #, similarity_matrix

def my_KL(alpha, c):
    alpha = alpha.cuda()
    beta = torch.ones((1, c)).cuda()  # c:batchsize
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)  
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def MyUncertianty(alpha,K_batchsize):
    alpha = alpha.cuda()
    L = torch.sum(alpha, dim=1, keepdim=True)
    # print("L:",L.shape)
    u = K_batchsize/L
    return u


def my_mse_loss(label, alpha, batch_size, lambda2):
    alpha = alpha.cuda()
    label = torch.eye(batch_size, batch_size).cuda()
    S = torch.sum(alpha, dim=1, keepdim=True)  #L
    E = alpha - 1
    m = alpha / S
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1  #128维
    C = lambda2 * my_KL(alp, batch_size)
    return (A + B) + C
    # return (A + B)   


def my_mse_loss_Notonlyone(original_label, alpha, batch_size, lambda2):  
    matrix_label = torch.zeros(batch_size,batch_size);
    for i in range(batch_size):
        t = original_label[i];
        for j in range(batch_size):
            if t==original_label[j]:
                matrix_label[i][j] = 1;
    alpha = alpha.cuda()
    label = matrix_label
    label = label.cuda()
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1  #128维
    C = lambda2 * my_KL(alp, batch_size)
    return (A + B) + C


def my_mse_loss_Notonlyone_distribute(original_label, alpha, batch_size, lambda2):  
    matrix_label = torch.zeros(batch_size,batch_size);
    for i in range(batch_size):
        t = original_label[i];
        for j in range(batch_size):
            if t == original_label[j]:
                matrix_label[i][j] = 1;
    # 把matrix_label变成分布的：0.0-1.0
    matrix_label = matrix_label / matrix_label.sum(dim=1)
    alpha = alpha.cuda()
    label = matrix_label
    label = label.cuda()
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1  #128维
    C = lambda2 * my_KL(alp, batch_size)
    return (A + B) + C





def new_edl_loss(func,original_label,alpha,class_num,batchsize):
    S = torch.sum(alpha, dim=1, keepdim=True)
    S = S.cuda()
    uncertainty = class_num / S   # 这个地方，ECCV和原来的代码有冲突，原来的是用batch_size/S。
    y = original_label.cuda()
    alpha = alpha.cuda()
    temp = 1 / alpha * y
    label_num = torch.sum(y, dim=1, keepdim=True)
    g = (1 - uncertainty.detach()) * label_num * torch.div(temp, torch.sum(temp, dim=1, keepdim=True))
    loss_clu = torch.sum(g * (func(S) - func(alpha)), dim=1, keepdim=True)
    # print("loss",loss_clu.shape,loss_clu)  # torch.Size([128, 1])
    loss_clu = torch.sum(loss_clu,dim=0)/batchsize
    return loss_clu

def new_my_edl_loss(func,original_label,alpha,class_num,batchsize):
    S = torch.sum(alpha, dim=1, keepdim=True)
    S = S.cuda()
    uncertainty = class_num / S   
    original_label = original_label.cuda()
    alpha = alpha.cuda()
    label_num = torch.sum(original_label, dim=1, keepdim=True)

    h = (1 - uncertainty.detach()) * label_num
    # loss_clu = torch.sum(h * (func(S) - func(alpha)), dim=1, keepdim=True)  
    # loss_clu = torch.sum(loss_clu,dim=0)/batchsize
    loss_clu =  (h * (func(S) - func(alpha))).mean()   

    # temp =  original_label
    # g = (1 - uncertainty.detach()) * label_num * torch.div(temp, torch.sum(temp, dim=1, keepdim=True))
    # loss_clu = torch.sum(g * (func(S) - func(alpha)), dim=1, keepdim=True)
    # loss_clu = torch.sum(loss_clu, dim=0) / batchsize
    # # print("loss",loss_clu.shape,loss_clu)  # torch.Size([128, 1])

    return loss_clu



def loglikelihood_loss(predict, target):  
    predict = predict.cuda()
    target = target.cuda()
    loglikelihood_err =  (predict - target) ** 2 
    loglikelihood_err = loglikelihood_err.cuda()
    return loglikelihood_err

def cross_loss(predict, target):
    loss = nn.CrossEntropyLoss(predict,target)
    return loss


# def new_ucom_loss():
def course_function_loss(epoch, total_epoch, batchsize, uncertain, loss_matrix, amplitude=0.7):  
    uncertain = uncertain.cuda()  # uncertain torch.Size([256, 1])
    idx = torch.arange(batchsize)  #
    theta = 2 * (idx + 0.5) / batchsize - 1  #第二项
    delta = - 2 * epoch / total_epoch + 1   #第一项
    curve = amplitude * torch.tanh(theta * delta) + 1  
    curve = curve.cuda()

    _, Uct_indexs = torch.sort(uncertain, dim=1) 

    Uct_indexs = Uct_indexs.cuda()
    Sorted_Curve = torch.zeros(batchsize).cuda()
    for i in range(batchsize):
        t = Uct_indexs[i]
        # print(f"T device: {t.device}")
        # print(f"curve device: {curve.device}")
        Sorted_Curve[i] = curve[t];    # todo SortCurve

    Sorted_Curve_matrix = torch.zeros(batchsize, batchsize).cuda()
    # print("Sorted_Curve_matrix", Sorted_Curve_matrix)
    for i in range(batchsize):
        Sorted_Curve_matrix[i] = Sorted_Curve[i].repeat((batchsize))
    #
    # print("bat",batchsize)
    # print("So",Sorted_Curve_matrix.shape)
    # print("loss",loss_matrix.shape)
    uct_guide_loss = torch.mul(Sorted_Curve_matrix, loss_matrix).mean()*10   
    return uct_guide_loss

def new_edl_kl_divergence(num_classes, alpha,batch_size): 
    alpha = alpha.cuda()
    beta = torch.ones([1, num_classes], dtype=torch.float32).to(alpha.device)  
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
          torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha) 
    dg1 = torch.digamma(alpha)

    # print("alpha",dg1)  #alpha torch.Size([128, 128])
    # print("beta",dg0)   #beta torch.Size([1, 10])
    beta = torch.ones([batch_size, batch_size], dtype=torch.float32).to(alpha.device)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    kl = kl.cuda()
    # print("kll:",kl.shape,list(kl))
    return kl








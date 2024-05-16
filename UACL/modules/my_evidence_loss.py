# -*-coding:utf-8-*-
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
# 证据学习(狄利克雷分布)的代码源：Deep Evidential Learning with Noisy Correspondence for  Cross-modal Retrieval

def compute_cosine_similarity(z_i,z_j,batch_size):
    cosine_similarity_matrix = np.zeros((batch_size,batch_size))
    # 打印相似度
    for i in range(batch_size):
        for j in range(batch_size):
            cosine_similarity_matrix[i][j] = torch.cosine_similarity(z_i[i], z_j[j], dim=0, eps=1e-08)  #调用torch的库，就不用再转置进行相乘了。
    # print("cosine_similarity_matrix:", cosine_similarity_matrix);
    cosine_similarity_matrix = torch.tensor(cosine_similarity_matrix)
    return cosine_similarity_matrix

def new_compute_cosine_similarity_2batchsize(z_i,z_j):
    z = torch.cat((z_i, z_j), dim=0)  # 吧z_i和z_j按行拼接起来（即行数增加，列数不变，数据“变长”）
    multiply = torch.matmul(z, z.T)  # 矩阵乘法
    cosine_similarity_matrix = torch.tensor(multiply)
    return cosine_similarity_matrix


def my_get_alpha(similarity_matrix, tau):
    similarity_matrix = torch.tensor(similarity_matrix);
    tau = torch.tensor(tau);
    evidences = torch.exp(torch.tanh(similarity_matrix) / tau)  # 与DECL也能对上
    # 打印 类似概率
    #print("evidences:",evidences);
    sum_e = evidences + evidences.t()
    norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)  # 归一化
    alpha_i2t = evidences + 1   #  对应DECL式8上面的绿色标注等式
    alpha_t2i = evidences.t() + 1
    # 打印
    # print("alpha_i2t的形状:", alpha_i2t.shape);    # alpha_i2t的形状: torch.Size([128, 128])
    # print("alpha_i2t:", alpha_i2t);   # 对角线数值：1.73e+3，其余地方最多+1
    #print("alpha_t2i:", alpha_t2i);
    # print("ecidnce：", evidences.shape, list(evidences))
    # print("alpha：", alpha_i2t.shape, list(alpha_i2t))
    sims_tanh = torch.tanh(similarity_matrix)   #  对应DECL的公式3
    return alpha_i2t, alpha_t2i, norm_e, sims_tanh #, similarity_matrix

def my_get_alpha_relu(similarity_matrix):
    similarity_matrix = torch.tensor(similarity_matrix);
    evidences = F.relu(similarity_matrix)
    sum_e = evidences + evidences.t()
    norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)  # 归一化
    alpha_i2t = evidences + 1
    alpha_t2i = evidences.t() + 1
    # sims_tanh = torch.tanh(similarity_matrix)
    return alpha_i2t, alpha_t2i, norm_e  #, sims_tanh #, similarity_matrix

def my_KL(alpha, c):
    alpha = alpha.cuda()
    beta = torch.ones((1, c)).cuda()  # c:batchsize
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)    # 应为查询相似度应该与概率对其，所以alpha二阶概率和不确定性建模
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

# todo 不需要伪标签，只把矩阵对角线视为正例时，调用这个函数
def my_mse_loss(label, alpha, batch_size, lambda2):  # label是真实标签。这里变为128*128方阵，  todo 只有对角线为1
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
    # return (A + B)   # 消融实验


# todo 当传进来的original_label是ground-truth的矩阵(值为所属类别的数字表示，e.g.1-10)时，调用这个函数
def my_mse_loss_Notonlyone(original_label, alpha, batch_size, lambda2):  # label这里变为128*128方阵，通过类别标签判断的相同类均为1
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

# todo 传进来的original_label是ground-truth的矩阵(值为所属类别的数字表示，e.g.1-10)时，并需要形成标签的分布时，调用这个函数
def my_mse_loss_Notonlyone_distribute(original_label, alpha, batch_size, lambda2):  # label这里变为128*128方阵，通过类别标签判断的相同类均为1，最后再求可能性的分布0-1
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


# ---------============----0626 发现之前使用了真实标签T^T 现在要用伪标签进行弥补-------=========---------#
# 这里是大于某个阈值，视为相似
# todo 不传标签进来，利用cos相似度进行判断，大于某个阈值，就把同簇标签置位为1。
def my_mse_loss_pseudo_Notonlyone(cosine_similarity_matrix, alpha, batch_size, lambda2):  # label这里变为128*128方阵，通过相似度进行伪标签计算，相似度大于阈值的相同类为1
    matrix_label = torch.zeros(batch_size,batch_size);
    for i in range(batch_size):
        for j in range(batch_size):
            if cosine_similarity_matrix[i][j] > 0.3:
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


def Labelof_frist_k_cosSimilirity(cosine_similarity_matrix,k,batch_size):
    cosine_similarity_matrix = torch.tensor(cosine_similarity_matrix)
    top_k_values,top_k_indexs = torch.topk(cosine_similarity_matrix,k,dim=1,largest=True)
    k_label = torch.zeros(batch_size,batch_size)
    for i in range(batch_size):
        for k in top_k_indexs[i]:
            k_label[i][k] = 1;
    return k_label

# 这里是前k个最大的相似度对应的索引数据，视为相似
# todo 不传标签进来，利用cos相似度进行判：对于每张图片，都把和它前K相似的标签置为1。
def my_mse_loss_pseudo_fristKNotonlyone(cosine_similarity_matrix, alpha, batch_size, lambda2):  # label这里变为128*128方阵，通过相似度进行伪标签计算，相似度大于阈值的相同类为1
    matrix_label = Labelof_frist_k_cosSimilirity(cosine_similarity_matrix,8,batch_size)  #  todo 12,6,8
    alpha = alpha.cuda()
    label = matrix_label.cuda()
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1  #128维
    C = lambda2 * my_KL(alp, batch_size)
    # print("A+B+C的形状",((A + B) + C).shape)   # A+B+C的形状 torch.Size([128, 1])
    # print("A+B+C",((A + B) + C))   # A+B+C的内容：tensor([[11.5602],[ 8.4726],...,])，应该是
    return (A + B) + C


# todo 当传进来的original_label是通过U得出的multi-hot矩阵时，调用这个函数
def my_mseloss_Uncen_Notonlyone(original_label, alpha, batch_size, lambda2):  # label这里变为128*128方阵， 通过传进来的标签判断，相同类赋值为1。
    alpha = alpha.cuda()
    label = original_label   # 当传进来的标签直接是multi-hot的矩阵时。
    label = label.cuda()
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1  #128维
    C = lambda2 * my_KL(alp, batch_size)
    return (A + B) + C

# todo 10.13 拓展期刊代码，这个函数尝试ECCV的DELU代码。 完全使用公式10
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

# todo 10.13 拓展期刊代码，尝试借助公式9，但把公式10里面的g替换成自己本来得到的multi-view标签original_label
def new_my_edl_loss(func,original_label,alpha,class_num,batchsize):
    S = torch.sum(alpha, dim=1, keepdim=True)
    S = S.cuda()
    uncertainty = class_num / S   # 这个地方，ECCV和原来的代码有冲突，原来的是用batch_size/S。
    original_label = original_label.cuda()
    alpha = alpha.cuda()
    label_num = torch.sum(original_label, dim=1, keepdim=True)

    # 直接用标签y的矩阵
    h = (1 - uncertainty.detach()) * label_num
    # loss_clu = torch.sum(h * (func(S) - func(alpha)), dim=1, keepdim=True)  # 1017之前的, 和下面一行一起运行，结果很大，有7k+
    # loss_clu = torch.sum(loss_clu,dim=0)/batchsize
    loss_clu =  (h * (func(S) - func(alpha))).mean()   # 使用这行代替上面两行，结果50+

    # 模仿公式10，对y进行归一化，但没有e参与
    # temp =  original_label
    # g = (1 - uncertainty.detach()) * label_num * torch.div(temp, torch.sum(temp, dim=1, keepdim=True))
    # loss_clu = torch.sum(g * (func(S) - func(alpha)), dim=1, keepdim=True)
    # loss_clu = torch.sum(loss_clu, dim=0) / batchsize
    # # print("loss",loss_clu.shape,loss_clu)  # torch.Size([128, 1])

    return loss_clu



def loglikelihood_loss(predict, target):  #为了使用 easy-hard，这里让其输出二维矩阵
    predict = predict.cuda()
    target = target.cuda()
    loglikelihood_err =  (predict - target) ** 2 # loglikelihood_err二维矩阵，batch_size*batch_size
    loglikelihood_err = loglikelihood_err.cuda()
    return loglikelihood_err

def cross_loss(predict, target):
    loss = nn.CrossEntropyLoss(predict,target)
    return loss


# def new_ucom_loss():
def course_function_loss(epoch, total_epoch, batchsize, uncertain, loss_matrix, amplitude=0.7):  # 先把uncer当做一维向量处理
    # 要求loss_matrix是形如loglikelihood_loss()的矩阵，其中每个位置记录着预测值和真实值之前的差异
    uncertain = uncertain.cuda()  # uncertain torch.Size([256, 1])
    idx = torch.arange(batchsize)  # 默认以0为起点，生成等差的一维张量
    theta = 2 * (idx + 0.5) / batchsize - 1  #第二项
    delta = - 2 * epoch / total_epoch + 1   #第一项
    curve = amplitude * torch.tanh(theta * delta) + 1  #curve的形状是(batchsize,)，它是一个一维张量
    curve = curve.cuda()

    _, Uct_indexs = torch.sort(uncertain, dim=1)  # uncertain应该是一个一维向量  升序排列 todo Sort_U_Index

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
    uct_guide_loss = torch.mul(Sorted_Curve_matrix, loss_matrix).mean()*10   # torch.mul就是对应位置相乘，torch.matmul是矩阵乘法.  .mean()在没有指定维度的情况下，就是对所有数进行求平均。


    return uct_guide_loss

def new_edl_kl_divergence(num_classes, alpha,batch_size):  #kl是交叉熵损失
    alpha = alpha.cuda()
    beta = torch.ones([1, num_classes], dtype=torch.float32).to(alpha.device)  # self.num_classes：类别数
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
          torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)  # 计算输入上 gamma 函数的对数导数
    dg1 = torch.digamma(alpha)

    # print("alpha",dg1)  #alpha torch.Size([128, 128])
    # print("beta",dg0)   #beta torch.Size([1, 10])
    beta = torch.ones([batch_size, batch_size], dtype=torch.float32).to(alpha.device)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    kl = kl.cuda()
    # print("kll:",kl.shape,list(kl))
    return kl








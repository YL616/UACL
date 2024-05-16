# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_sdm(image_fetures, text_fetures,batch_size, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    # batch_size = image_fetures.shape[0];
    #pid = pid.cuda();
    #pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    #pid_dist = pid - pid.t()
    #labels = (pid_dist == 0).float()

    matrix_label = torch.zeros(batch_size, batch_size).cuda();
    for i in range(batch_size):
        t = pid[i];
        for j in range(batch_size):
            if t == pid[j]:
                matrix_label[i][j] = 1;
    labels = matrix_label.cuda()
    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    # print('img:', image_norm.shape)    # [128,128]
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)
    image_norm = image_norm.cuda();
    text_norm = text_norm.cuda();

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()
    t2i_cosine_theta = t2i_cosine_theta.cuda();
    i2t_cosine_theta = i2t_cosine_theta.cuda();

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta
    text_proj_image = text_proj_image.cuda();
    image_proj_text = image_proj_text.cuda();

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)
    # labels_distribute = labels;

    i2t_pred = F.softmax(image_proj_text.cuda(), dim=1)
    i2t_loss = i2t_pred.cuda() * (F.log_softmax(image_proj_text.cuda(), dim=1) - torch.log(labels_distribute.cuda() + epsilon))
    t2i_pred = F.softmax(text_proj_image.cuda(), dim=1)
    t2i_loss = t2i_pred.cuda() * (F.log_softmax(text_proj_image.cuda(), dim=1) - torch.log(labels_distribute.cuda() + epsilon))

    loss = torch.mean(torch.sum(i2t_loss.cuda(), dim=1)).cuda() + torch.mean(torch.sum(t2i_loss.cuda(), dim=1)).cuda()

    return loss


def compute_sdm_Uncertain(image_fetures, text_fetures,batch_size, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    # 写论文的时候，这里其实是把特征和指示矩阵对齐的（利用相似度矩阵+uncertainty的筛选，推导得出的指示矩阵）
    """
    Similarity Distribution Matching
    """
    # batch_size = image_fetures.shape[0];
    #pid = pid.cuda();
    #pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    #pid_dist = pid - pid.t()
    #labels = (pid_dist == 0).float()
    # matrix_label = torch.zeros(batch_size, batch_size).cuda();
    # for i in range(batch_size):
    #     t = pid[i];
    #     for j in range(batch_size):
    #         if t == pid[j]:
    #             matrix_label[i][j] = 1;
    # labels = matrix_label.cuda()

    labels = pid.cuda()
    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    # print('img:', image_norm.shape)    # [128,128]
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)
    image_norm = image_norm.cuda();
    text_norm = text_norm.cuda();

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()
    t2i_cosine_theta = t2i_cosine_theta.cuda();
    i2t_cosine_theta = i2t_cosine_theta.cuda();

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta
    text_proj_image = text_proj_image.cuda();
    image_proj_text = image_proj_text.cuda();

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)
    # labels_distribute = labels;

    i2t_pred = F.softmax(image_proj_text.cuda(), dim=1)
    i2t_loss = i2t_pred.cuda() * (F.log_softmax(image_proj_text.cuda(), dim=1) - torch.log(labels_distribute.cuda() + epsilon))
    t2i_pred = F.softmax(text_proj_image.cuda(), dim=1)
    t2i_loss = t2i_pred.cuda() * (F.log_softmax(text_proj_image.cuda(), dim=1) - torch.log(labels_distribute.cuda() + epsilon))

    loss = torch.mean(torch.sum(i2t_loss.cuda(), dim=1)).cuda() + torch.mean(torch.sum(t2i_loss.cuda(), dim=1)).cuda()

    return loss

def compute_pseudo_sdm(image_fetures, text_fetures,batch_size, cosine_similarity_matrix, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    matrix_label = torch.zeros(batch_size, batch_size).cuda();
    for i in range(batch_size):
        for j in range(batch_size):
            if cosine_similarity_matrix[i][j] > 0.3:
                matrix_label[i][j] = 1;
    labels = matrix_label.cuda()
    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    # print('img:', image_norm.shape)    # [128,128]
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)
    image_norm = image_norm.cuda();
    text_norm = text_norm.cuda();

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()
    t2i_cosine_theta = t2i_cosine_theta.cuda();
    i2t_cosine_theta = i2t_cosine_theta.cuda();

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta
    text_proj_image = text_proj_image.cuda();
    image_proj_text = image_proj_text.cuda();

    # normalize the true matching distribution
    #labels_distribute = labels / labels.sum(dim=1)
    labels_distribute = labels;

    i2t_pred = F.softmax(image_proj_text.cuda(), dim=1)
    i2t_loss = i2t_pred.cuda() * (F.log_softmax(image_proj_text.cuda(), dim=1) - torch.log(labels_distribute.cuda() + epsilon))
    t2i_pred = F.softmax(text_proj_image.cuda(), dim=1)
    t2i_loss = t2i_pred.cuda() * (F.log_softmax(text_proj_image.cuda(), dim=1) - torch.log(labels_distribute.cuda() + epsilon))

    loss = torch.mean(torch.sum(i2t_loss.cuda(), dim=1)).cuda() + torch.mean(torch.sum(t2i_loss.cuda(), dim=1)).cuda()
    #print('loss:', loss);
    return loss

def compute_pseudo_fristk_sdm(image_fetures, text_fetures,batch_size, cosine_similarity_matrix, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    # 写论文的时候，这里其实是把特征和指示矩阵对齐的（利用相似度矩阵，推导得出的指示矩阵）
    matrix_label = Labelof_frist_k_cosSimilirity(cosine_similarity_matrix,8,batch_size)   #  todo 12,6,8
    # matrix_label = torch.zeros(batch_size, batch_size).cuda();
    # for i in range(batch_size):
    #     for j in range(batch_size):
    #         if cosine_similarity_matrix[i][j] > 0.3:
    #             matrix_label[i][j] = 1;
    labels = matrix_label.cuda()
    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    # print('img:', image_norm.shape)    # [128,128]
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)
    image_norm = image_norm.cuda();
    text_norm = text_norm.cuda();

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()
    t2i_cosine_theta = t2i_cosine_theta.cuda();
    i2t_cosine_theta = i2t_cosine_theta.cuda();

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta
    text_proj_image = text_proj_image.cuda();
    image_proj_text = image_proj_text.cuda();

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)
    # 0711之前，直接用的labels(下面的语句)，没有将其归一化(上面的语句)
    # labels_distribute = labels;

    i2t_pred = F.softmax(image_proj_text.cuda(), dim=1)
    i2t_loss = i2t_pred.cuda() * (F.log_softmax(image_proj_text.cuda(), dim=1) - torch.log(labels_distribute.cuda() + epsilon))
    t2i_pred = F.softmax(text_proj_image.cuda(), dim=1)
    t2i_loss = t2i_pred.cuda() * (F.log_softmax(text_proj_image.cuda(), dim=1) - torch.log(labels_distribute.cuda() + epsilon))

    loss = torch.mean(torch.sum(i2t_loss.cuda(), dim=1)).cuda() + torch.mean(torch.sum(t2i_loss.cuda(), dim=1)).cuda()
    #print('loss:', loss);
    return loss


def Labelof_frist_k_cosSimilirity(cosine_similarity_matrix,k,batch_size):
    cosine_similarity_matrix = torch.tensor(cosine_similarity_matrix)
    top_k_values,top_k_indexs = torch.topk(cosine_similarity_matrix,k,dim=1,largest=True)
    k_label = torch.zeros(batch_size,batch_size)
    for i in range(batch_size):
        for k in top_k_indexs[i]:
            k_label[i][k] = 1;
    return k_label

# def k_print(cosine_similarity_matrix,k):
#     cosine_similarity_matrix = torch.tensor(cosine_similarity_matrix)
#     top_k_values, top_k_indexs = torch.topk(cosine_similarity_matrix, k, dim=1, largest=True)
#     path = "./k.txt";
#     with open(path, 'a') as f:
#         f.write(str(top_k_values) + '\n');
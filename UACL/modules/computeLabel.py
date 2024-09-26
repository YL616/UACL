# -*-coding:utf-8-*-
import torch
import numpy as np
import torch.nn.functional as F


def deleteHigh_Uncertainty(cosine_similarity_matrix,uncertainty,batch_size,k_similar,k_uncertrain,k_cut):
    cosine_similarity_matrix = torch.tensor(cosine_similarity_matrix)
    top_k_values,top_k_indexs = torch.topk(cosine_similarity_matrix,k_similar,dim=1,largest=True)
    k_simi_label = torch.zeros(batch_size,batch_size)    
    for i in range(batch_size):
        for k in top_k_indexs[i]:
            k_simi_label[i][k] = 1;

    top_k_values, top_k_indexs = torch.topk(uncertainty, k_uncertrain,dim=0,largest=True)

    k_uncer_label = torch.zeros(batch_size)
    for k in top_k_indexs:
        k_uncer_label[k] = 1;

    union_label = torch.zeros(batch_size,batch_size)
    for i in range(batch_size):
        if k_uncer_label[i]==1:  
            union_label[i] = cut_one(k_simi_label[i],k_cut,batch_size)
        else:
            union_label[i] = k_simi_label[i]

    return union_label

def cut_one(original_label,k_cut,batch_size):
    # original_label = torch.zeros(batch_size)
    top_k_values, top_k_indexs = torch.topk(original_label, k_cut, largest=True)
    result =  torch.zeros(batch_size)
    for k in top_k_indexs:
        result[k] = 1;
    return result


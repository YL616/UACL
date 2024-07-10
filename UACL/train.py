import numpy as np
import torch
from modules import transform, resnet, network, contrastive_loss,my_evidence_loss,my_sdm_loss,computeLabel,DELU_loss,ECCV_loss
from torch.nn.functional import normalize
import os
from torch.utils.tensorboard import SummaryWriter


# os.environ["CUDA_VISIBLE_DEVICES"] = "3";

def train_net(net, data_loader, optimizer, batch_size, zeta,epoch,max_epochs,num_class):
    net.train()
    for param in net.parameters():
        param.requires_grad = True
    # writer = SummaryWriter("logs")

    loss = 0;
    for step, ((x_i, x_j), label) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')   # x_i:torch.Size([64, 3, 224, 224])
        x_j = x_j.to('cuda')
        # print("x_i", x_i.shape, x_i)
        # print("x_j", x_j.shape, x_j)
        h_i = net.resnet(x_i)     #
        h_j = net.resnet(x_j)

        # writer.add_graph(net, (x_i, x_j))
        # writer.close()
        z_i = normalize(net.instance_projector(h_i), dim=1)
        z_j = normalize(net.instance_projector(h_j), dim=1)

        c_i = net.cluster_projector(h_i)
        c_j = net.cluster_projector(h_j)
        # print("z_i", z_i.shape, z_i)
        # print("z_j", z_j.shape, z_j)

#         loss_true = contrastive_loss.truelabel_my_one_hot_edl(c_i,c_j,epoch,num_class,batch_size,label)

        # loss1 = contrastive_loss.C3_loss(z_i, z_j, batch_size, zeta)
        # loss1_1 = contrastive_loss.CC_Cluster_loss(num_class,c_i,c_j)
        loss1_2 = contrastive_loss.my_one_hot_edl(c_i,c_j,epoch,num_class,batch_size)  #
        loss2 = ECCV_loss.my_ECCV_loss(z_i,z_j,batch_size,num_class,torch.log,epoch,max_epochs)

        # loss = loss1
        # loss = loss1_1
        # loss = loss1+loss1_2
        # loss = loss2
        # loss = loss1_2
        loss =  loss1_2+loss2
        # loss = loss1+loss1_1
        # loss = loss_true+loss1
        # loss2.requires_grad_(True)
        if step == 0:
            # print("loss_C3", loss1)  # loss_C3 is_leaf:False grad_fn:<DivBackward0 object at 0x7fa989029850>
            # print("loss_CC_clu", loss1_1)  # loss_C3 is_leaf:False grad_fn:<DivBackward0 object at 0x7fa989029850>
            print("loss_one_hot", loss1_2)  # tensor(0.8934, device='cuda:0', grad_fn=<MeanBackward0>)
            # print("loss_one_hot", loss_true)  #
            print("loss_ECCV", loss2)  # loss_clu is_leaf:True grad_fn:None
            # print("loss_total", loss)  # loss_total False <AddBackward0 object at 0x7fa989029850>

        loss.requires_grad_(True)
        loss.backward()
        # if step == 0:
        #     for name, param in net.named_parameters():
        #         if param.grad is None:
        #             print(name)

        optimizer.step()


    print("loss:", loss);


    return net , optimizer

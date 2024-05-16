import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
#nn.Sequential是用于构建神经网络的PyTorch模块，允许用户按照顺序将多个层组合在一起，以构建一个神经网络模型
        # fc_layer = nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim)   # 使用同一个层，共享权重之后反而效果很差（C3就已经很差了）
        self.instance_projector = nn.Sequential(
            # nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),  # Linear(in_features=512, out_features=512, bias=True)
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),    #  Linear(in_features=512, out_features=128, bias=True)
        )
        self.cluster_projector = nn.Sequential(
            # nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim), # Linear(in_features=512, out_features=512, bias=True)
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num), # Linear(in_features=512, out_features=10, bias=True)
            nn.Softmax(dim=1)
            # nn.ReLU()
        )

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)  # 得到特征提取的结果
        # print('repdim:', self.resnet.rep_dim)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)  # 对特征提取结果进行归一化？ 对，应该是样例级别，得到所属类别可能性后，需要归一化（一张图就属于一个类）

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)  #

        return z_i, z_j, c_i, c_j


    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1) # torch.argmax是一个PyTorch函数，它返回给定维度上张量中最大值的索引
        #将返回c张量中每行最大值的索引，这些索引将被用作聚类的标签。
        return c  # c是x最有可能所属的类别？

import os
import numpy as np
import torch
import torchvision
import argparse
from collections import OrderedDict

from modules import transform, resnet, network
from utils import yaml_config_hook
from torch.utils import data
import torch.utils.data.distributed
from evaluation import evaluation
from train import train_net
import selectUn_train
import DELUloss_train
import sFeature_train
import train_layerResNet
import train

os.environ["CUDA_VISIBLE_DEVICES"] = "1";

def main():
    parser = argparse.ArgumentParser()  #创建一个解析器对象，用于解析命令行参数
    config = yaml_config_hook.yaml_config_hook("config/config.yaml") #从config.yaml文件中读取配置信息，

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v)) #将配置文件中的参数添加到命令行解析器。
    # 对于配置文件中的每个键值对，都会添加一个命令行参数，其中参数名为键，参数默认值为值，参数类型为值的类型。这样，用户可以在命令行中通过指定参数来覆盖配置文件中的默认值。例如，如果配置文件中的参数为learning_rate: 0.001，但用户想要使用不同的学习率，可以在命令行中使用--learning_rate 0.01来指定新的学习率
    args = parser.parse_args() #将命令行参数解析为Python对象，并将其存储在args变量中。parse_args()方法解析命令行参数并返回一个命名空间，其中包含所有的参数值。这些参数可以通过args变量进行访问
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)   #设置随机种子。 作用是什么？哪里用得到随机种子？
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data---------------------------------------------------------------------------------------------------------------------------------------------------
    #train data
    # 进行了数据增强操作（s应该是缩放系数？）：“=0.5表示对图像进行随机缩放，缩放因子在[1-s, 1+s]之间”
# 一、CIFAR-10数据集
#     train_dataset = torchvision.datasets.CIFAR10(
#             root=args.dataset_dir,
#             download=True,
#             train=True,
#             transform=transform.Transforms(size=args.image_size, s=0.5),
#         )
#
#     test_dataset = torchvision.datasets.CIFAR10(
#             root=args.dataset_dir,
#             download=True,
#             train=False,
#             transform=transform.Transforms(size=args.image_size, s=0.5),
#         )
#     dataset = data.ConcatDataset([train_dataset, test_dataset])   #dataset：把CIFAR10数据集的train和test，数据增强后合在一起，当训练集？
#     # 相当于dataset是一个封装的结果，train_dataset和test_dataset是两个独立的整体，被封装在dataset中，当想要dataset[0]时，给出tensor:train_dataset[0]，tensor:test_dataset[0]
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
#                                               pin_memory=True)     #按照128分batch
#     # CIFAR10数据集的test
#     test_dataset_1 = torchvision.datasets.CIFAR10(
#         root=args.dataset_dir,
#         download=True,
#         train=True,
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     test_dataset_2 = torchvision.datasets.CIFAR10(
#         root=args.dataset_dir,
#         download=True,
#         train=False,
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     dataset_test = data.ConcatDataset([test_dataset_1, test_dataset_2])  # 把CIFAR10数据集的train和test，简单预处理(剪裁大小)后合在一起，当测试集？
#     test_loader = torch.utils.data.DataLoader(
#         dataset=dataset_test,
#         batch_size=args.test_batch_size,
#         shuffle=False)  # 按照500分batch


#  二、 CIFAR-100 数据集
#     train_dataset = torchvision.datasets.CIFAR100(
#             root=args.dataset_dir,
#             download=True,
#             train=True,
#             transform=transform.Transforms(size=args.image_size, s=0.5),
#         )
#     test_dataset = torchvision.datasets.CIFAR100(
#             root=args.dataset_dir,
#             download=True,
#             train=False,
#             transform=transform.Transforms(size=args.image_size, s=0.5),
#         )
#     dataset = data.ConcatDataset([train_dataset, test_dataset])   #dataset：把CIFAR10数据集的train和test，数据增强后合在一起，当训练集？
#     # 相当于dataset是一个封装的结果，train_dataset和test_dataset是两个独立的整体，被封装在dataset中，当想要dataset[0]时，给出tensor:train_dataset[0]，tensor:test_dataset[0]
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
#                                               pin_memory=True)     #按照128分batch
#
#     test_dataset_1 = torchvision.datasets.CIFAR100(
#         root=args.dataset_dir,
#         download=True,
#         train=True,
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     test_dataset_2 = torchvision.datasets.CIFAR100(
#         root=args.dataset_dir,
#         download=True,
#         train=False,
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     dataset_test = data.ConcatDataset([test_dataset_1, test_dataset_2])  # 把CIFAR10数据集的train和test，简单预处理(剪裁大小)后合在一起，当测试集？
#     test_loader = torch.utils.data.DataLoader(
#         dataset=dataset_test,
#         batch_size=args.test_batch_size,
#         shuffle=False)  # 按照500分batch


#  #  三、ImgNet-10数据集
#     train_dataset = torchvision.datasets.ImageFolder(
#         root='./imagenet-10',
#         transform=transform.Transforms(size=args.image_size, blur=True),
#     )
#     data_loader = torch.utils.data.DataLoader(
#         dataset=train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         drop_last=True)
#
#     dataset_test = torchvision.datasets.ImageFolder(
#         root='./imagenet-10',
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     test_loader = torch.utils.data.DataLoader(
#         dataset=dataset_test,
#         batch_size=args.test_batch_size,
#         shuffle=False,
#         drop_last=True)


#  四、tiny-ImageNet数据集
#     train_dataset = torchvision.datasets.ImageFolder(
#         root='./dataset/tiny-imagenet-200/train',
#         transform=transform.Transforms(s=0.5,size=args.image_size),
#     )
#     data_loader = torch.utils.data.DataLoader(
#         dataset=train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         drop_last=True)
#
#     dataset_test = torchvision.datasets.ImageFolder(
#         root='./dataset/tiny-imagenet-200/train',
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     test_loader = torch.utils.data.DataLoader(
#         dataset=dataset_test,
#         batch_size=args.test_batch_size,
#         shuffle=False,
#         drop_last=False)

#    五、STL-10数据集
#     train_dataset = torchvision.datasets.STL10(
#         root=args.dataset_dir,
#         split="train",
#         download=True,
#         transform=transform.Transforms(size=args.image_size),
#     )
#     test_dataset = torchvision.datasets.STL10(
#         root=args.dataset_dir,
#         split="test",
#         download=True,
#         transform=transform.Transforms(size=args.image_size),
#     )
#     cluster_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
#     data_loader = torch.utils.data.DataLoader(
#         cluster_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         drop_last=True,
#         num_workers=args.workers,
#     )
# #
#     train_dataset1 = torchvision.datasets.STL10(
#         root=args.dataset_dir,
#         split="train",
#         download=True,
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     test_dataset2 = torchvision.datasets.STL10(
#         root=args.dataset_dir,
#         split="test",
#         download=True,
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     dataset = torch.utils.data.ConcatDataset([train_dataset1, test_dataset2])
#     test_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=500,
#         shuffle=False,
#         drop_last=False,
#         num_workers=args.workers,
#     )

# #   六、ImageNet_Dogs数据集
    dataset = torchvision.datasets.ImageFolder(
        root='dataset/imagenet-dogs',
        transform=transform.Transforms(size=args.image_size, blur=True),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    dataset1 = torchvision.datasets.ImageFolder(
        root='dataset/imagenet-dogs',
        transform=transform.Transforms(size=args.image_size).test_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset1,
        batch_size=500,
        shuffle=False,    #shuffle是先打乱，再取batch
        drop_last=False,
        num_workers=args.workers,
    )



#     # Initializing our network with a network trained with CC 初始化一个网络！-------------------------------------------------------------------------------------------------------
    res = resnet.get_resnet(args.resnet)   # 需要哪个resnet就把哪个的名字传进去
    net = network.Network(res, args.feature_dim, args.class_num)
    net = net.to('cuda')
    # print(net)



    # 在下面这个程序块中，预训练模型的权重被加载到名为checkpoint的字典中。然后，使用一个循环将checkpoint中的权重复制到new_state_dict中。
# 在复制之前，将每个键名中的module.前缀删除。最后，使用net.load_state_dict(new_state_dict)将new_state_dict中的权重加载到神经网络net中。这个过程的目的是将预训练模型的权重加载到我们的神经网络中，以便我们可以使用这些权重来初始化我们的网络
   # 原代码，读取的是CIFAR_10的原始参数值：
#     checkpoint = torch.load('CIFAR_10_initial', map_location=torch.device('cuda:0')) # 这个应该是C3给提供的
#     checkpoint = torch.load('CCgive_CIFAR10-checkpoint_1000.tar', map_location=torch.device('cuda:0'));  # 0.789，最优的初始化参数集！ 10-checkpoint_1000.tar是CC代码里给出的CIFAR10数据集训练1000epoch的参数
#     checkpoint = torch.load('CCgive1102CIFAR10check_1000.tar', map_location=torch.device('cuda:0'));  #CC代码里直接给的cifar100参数，跑出来epoch0是0.170
#     checkpoint = torch.load('cifar100-runCC-checkpoint_1000.tar', map_location=torch.device('cuda:0'));  #cifar100-runCC-checkpoint_1000.tar 自己跑CC代码得出的cifar100数据集的1000epoch参数
#     checkpoint = torch.load('runCC_Imagenet10-checkpoint_1000.tar', map_location=torch.device('cuda:0'));  #10-checkpoint_1000.tar是CC代码里给出的CIFAR10数据集训练1000epoch的参数
#     checkpoint = torch.load('STL-10_CCgive_checkpoint_1000.tar', map_location=torch.device('cuda:0'));  #CIFAR100_CCgive-checkpoint_1000.tar是CC代码里直接给出的CIFAR100数据集训练1000epoch的参数
#     checkpoint = torch.load('Tiny-ImageNet_runCC_checkpoint_800.tar', map_location=torch.device('cuda:0'));  #Tiny-ImageNet_runCC_checkpoint_800.tar 用CC代码跑Tiny-ImageNet数据集，第800步的参数结果
#     checkpoint = torch.load('Tiny-ImageNet_runCC_checkpoint_1000.tar', map_location=torch.device('cuda:0'));  #Tiny-ImageNet_runCC_checkpoint_1000.tar 用CC代码跑Tiny-ImageNet数据集，第1000步的参数结果
    checkpoint = torch.load('runCC_image_dogs_checkpoint_1000.tar', map_location=torch.device('cuda:0'));  #runCC_image_dogs_checkpoint_1000.tar 用CC代码跑ImageNet-Dogs数据集，第1000步的参数结果
#     checkpoint = torch.load('CIFAR_10_C3_loss_epoch_19', map_location=torch.device('cuda:0'));

    new_state_dict = OrderedDict()   #创建一个有序字典，用于存储从预训练模型中加载的权重。

# 从C3readme中下载的CIFAR_initial参数处理过程
#     for k, v in checkpoint['net'].items():
#         name = k[7:]  # remove `module.`
#         new_state_dict[name] = v
#     net.load_state_dict(new_state_dict)

# 从C3原代码中直接down的checkpoint_1000参数处理过程
    for k, v in checkpoint['net'].items():
        new_state_dict[k] = v
    net.load_state_dict(new_state_dict)

    # optimizer ---------------------------------------------------------------------------------------------------------------------------------------------
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    # train loop ---------------------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(args.start_epoch, args.max_epochs):

        print("epoch:", epoch)
        evaluation.net_evaluation(net,test_loader,args.dataset_size, args.test_batch_size)  #先得到ACC等参数

        # 计算初始化参数(CC的结果)对应的Feature和label
        # if epoch == 0:
        #     feature, pre_label, true_label = evaluation.get_predict_label(net, test_loader, args.dataset_size,args.test_batch_size)
        #     a = feature.detach().numpy()
        #     path = "./tSNE/tSEN_feature";   #特征只有保存一份
        #     txt = path + str(epoch)
        #     txt = txt + '.txt'
        #     np.savetxt(txt, a)
        #
        #     path_ture = "./tSNE/label_ture";   #标签要保存两份
        #     txt = path_ture + str(epoch)
        #     txt = txt + '.txt'
        #     np.savetxt(txt, true_label)
        #
        #     path_pre = "./tSNE/label_predict";  # 标签要保存两份
        #     txt = path_pre + str(epoch)
        #     txt = txt + '.txt'net

        #     np.savetxt(txt, pre_label)
        #     print("保存了第" + str(epoch) + "个epoch数据")

#         net, optimizer = train_net(net, data_loader, optimizer, args.batch_size, args.zeta)  #原C3的train过程。

# # 把不确定性低的筛选出去  todo old ！！！
#         net, optimizer = selectUn_train.train_net(net, data_loader, optimizer, args.batch_size, args.zeta,epoch,args.class_num)


# 2023  10  月  扩展期刊的损失函数 todo new ！！！
#         net, optimizer = DELUloss_train.train_net(net, data_loader, optimizer, args.batch_size, args.zeta,epoch,args.max_epochs,args.class_num)

        net, optimizer = train.train_net(net, data_loader, optimizer, args.batch_size, args.zeta, epoch, args.max_epochs, args.class_num)




# 得到中间处理过程，数据降维的结果（但后面直接在evaluation.get_predict_label中一起得到特征、预测标签和真实标签，所以这个结果就没用了
#         net, optimizer = sFeature_train.train_net(net, data_loader, optimizer, args.batch_size, args.zeta)
        feature, pre_label, true_label = evaluation.get_predict_label(net, test_loader, args.dataset_size, args.test_batch_size)
        # print(list(feature))
        # if epoch==10 or epoch==19:
        #     a = feature.detach().numpy()
        #     path = "./tSNE/tSEN_feature";
        #     txt = path+str(epoch)
        #     txt = txt + '.txt'
        #     np.savetxt(txt, a)
        #
        #     path_ture = "./tSNE/label_ture";  # 标签要保存两份
        #     txt = path_ture + str(epoch)
        #     txt = txt + '.txt'
        #     np.savetxt(txt, true_label)
        #
        #     path_pre = "./tSNE/label_predict";  # 标签要保存两份
        #     txt = path_pre + str(epoch)
        #     txt = txt + '.txt'
        #     np.savetxt(txt, pre_label)
        #
        #     print("保存了第"+str(epoch)+"个epoch数据")




    # 将resNet分层，得到不同粒度的hi数据，直接传入loss，进行结果的直接相加
        # net, optimizer = train_layerResNet.train_net(net, data_loader, optimizer, args.batch_size, args.zeta);

    # 先不把每一步的参数都写入
    #     if epoch>=500 and epoch%100==0:
    #         state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    #         with open('CIFAR_100_C3_loss_epoch_{}'.format(epoch), 'wb') as out:
    #             torch.save(state, out)


if __name__ == "__main__":
    main()

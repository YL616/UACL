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
    parser = argparse.ArgumentParser()  
    config = yaml_config_hook.yaml_config_hook("config/config.yaml")

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v)) 
   
    args = parser.parse_args() 
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)  
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data---------------------------------------------------------------------------------------------------------------------------------------------------
    #train data
   
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
#     dataset = data.ConcatDataset([train_dataset, test_dataset])   
#     
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
#                                               pin_memory=True)     
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
#     dataset = data.ConcatDataset([train_dataset, test_dataset])   
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
#     dataset_test = data.ConcatDataset([test_dataset_1, test_dataset_2])  
#     test_loader = torch.utils.data.DataLoader(
#         dataset=dataset_test,
#         batch_size=args.test_batch_size,
#         shuffle=False)  # 按照500分batch



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




    checkpoint = torch.load('runCC_image_dogs_checkpoint_1000.tar', map_location=torch.device('cuda:0'));  #runCC_image_dogs_checkpoint_1000.tar 用CC代码跑ImageNet-Dogs数据集，第1000步的参数结果

    new_state_dict = OrderedDict()   

#     for k, v in checkpoint['net'].items():
#         name = k[7:]  # remove `module.`
#         new_state_dict[name] = v
#     net.load_state_dict(new_state_dict)


    for k, v in checkpoint['net'].items():
        new_state_dict[k] = v
    net.load_state_dict(new_state_dict)

    # optimizer ---------------------------------------------------------------------------------------------------------------------------------------------
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    # train loop ---------------------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(args.start_epoch, args.max_epochs):

        print("epoch:", epoch)
        evaluation.net_evaluation(net,test_loader,args.dataset_size, args.test_batch_size)  


        # if epoch == 0:
        #     feature, pre_label, true_label = evaluation.get_predict_label(net, test_loader, args.dataset_size,args.test_batch_size)
        #     a = feature.detach().numpy()
        #     path = "./tSNE/tSEN_feature";   
        #     txt = path + str(epoch)
        #     txt = txt + '.txt'
        #     np.savetxt(txt, a)
        #
        #     path_ture = "./tSNE/label_ture"; 
        #     txt = path_ture + str(epoch)
        #     txt = txt + '.txt'
        #     np.savetxt(txt, true_label)
        #
        #     path_pre = "./tSNE/label_predict";  
        #     txt = path_pre + str(epoch)
        #     txt = txt + '.txt'net

        #     np.savetxt(txt, pre_label)
        #     print("保存了第" + str(epoch) + "个epoch数据")

#         net, optimizer = train_net(net, data_loader, optimizer, args.batch_size, args.zeta) 

#         net, optimizer = selectUn_train.train_net(net, data_loader, optimizer, args.batch_size, args.zeta,epoch,args.class_num)


#         net, optimizer = DELUloss_train.train_net(net, data_loader, optimizer, args.batch_size, args.zeta,epoch,args.max_epochs,args.class_num)

        net, optimizer = train.train_net(net, data_loader, optimizer, args.batch_size, args.zeta, epoch, args.max_epochs, args.class_num)





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
        #     path_ture = "./tSNE/label_ture";  
        #     txt = path_ture + str(epoch)
        #     txt = txt + '.txt'
        #     np.savetxt(txt, true_label)
        #
        #     path_pre = "./tSNE/label_predict";  
        #     txt = path_pre + str(epoch)
        #     txt = txt + '.txt'
        #     np.savetxt(txt, pre_label)
        #
        #     print("保存了第"+str(epoch)+"个epoch数据")



        # net, optimizer = train_layerResNet.train_net(net, data_loader, optimizer, args.batch_size, args.zeta);

    #     if epoch>=500 and epoch%100==0:
    #         state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    #         with open('CIFAR_100_C3_loss_epoch_{}'.format(epoch), 'wb') as out:
    #             torch.save(state, out)


if __name__ == "__main__":
    main()

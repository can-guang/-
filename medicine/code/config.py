# -*- coding: UTF-8 -*-
import torch
TEST_SPLIT_INDEX = 1

class Config(object):
    def config():
        ARGS = {}
        ARGS['device'] = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        ARGS['model_save_dir'] = 'model_save_dir_gcn50_fusing'  #模型保存文件夹
        ARGS['image_dir_train'] = '/home/ruancanguang/medicine/medicine/data/data_train_new/'
        ARGS['label_train_file'] = 'train_dict.npy'
        ARGS['image_dir_test'] = '/home/ruancanguang/medicine/medicine/data/data_test_new/'
        ARGS['label_test_file'] = 'test_dict.npy'
        ARGS['log_dir'] = 'logs_resnet_gcn50'  #损失图保存文件夹
        ARGS['type'] = torch.FloatTensor
        ARGS['is_train'] = True
        ARGS['epoch'] = 100
        ARGS['model_save_epoch'] = 1
        ARGS['batch_size'] = 10
        ARGS['label_nums'] = 147
        return ARGS





import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import config
from utils import read_image



class TorchDataset(Dataset):
    def __init__(self, image_path, label_file, resize_height=224, resize_width=224, repeat=1, is_train=True):
        '''
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.image_dir = image_path
        self.label_file = label_file
        self.file_list = os.listdir(self.image_dir)
        self.len = len(self.file_list)
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.repeat = repeat
        self.is_train = is_train


        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
        self.toTensor = transforms.ToTensor()

        '''class torchvision.transforms.Normalize(mean, std)
        此转换类作用于torch. * Tensor,给定均值(R, G, B) 和标准差(R, G, B)，
        用公式channel = (channel - mean) / std进行规范化。
        '''
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, i):
        index = i % self.len
        image_name = self.file_list[index]
        image_index = image_name[:-4]
        image_index = int(image_index)
        image_path = os.path.join(self.image_dir, image_name)
        img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=True)
        img = self.data_preproccess(img)
        img = img.to(torch.float32)
        if self.is_train:
            dict = np.load(self.label_file, allow_pickle=True).item()
            label = dict[image_index]
            label_list = label.tolist()
            label_list = torch.tensor(label_list)
        else:
            dict = np.load(self.label_file, allow_pickle=True).item()
            label = dict[image_index]
            label_list = label.tolist()
            label_list = torch.tensor(label_list)
        img = img.numpy()
        return img, label_list

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.file_list) * self.repeat
        return data_len


    def load_data(self, path, resize_height, resize_width, normalization ):
        '''
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        '''
        image = read_image(path, resize_height, resize_width, normalization)
        return image

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        data = self.toTensor(data)
        return data


def load_data(is_train=True):
    arg = config.Config.config()
    image_dir_train = arg['image_dir_train']
    image_dir_test = arg['image_dir_test']
    batch_size = arg['batch_size']
    label_train_file = arg['label_train_file']
    label_test_file = arg['label_test_file']

    if is_train:
        data_loder = TorchDataset(image_path=image_dir_train, label_file=label_train_file, is_train=True)
        data_loder = DataLoader(dataset=data_loder, batch_size=batch_size, shuffle=False)
    else:
        data_loder = TorchDataset(image_path=image_dir_test, label_file=label_test_file, is_train=False)
        data_loder = DataLoader(dataset=data_loder, batch_size=batch_size, shuffle=False)

    return data_loder


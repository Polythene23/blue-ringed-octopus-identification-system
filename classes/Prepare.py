import math
import os
import random
import shutil
from PIL import Image
import os
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import pickle

class Prepare:
    i = 1

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.train_path = ''
        self.test_path = ''

    def start(self):
        # 1. 划分数据集，得到数据用于表格和可视化（图片数目统计）
        cates,data = self.divideImages()
        # 2. 数据可视化
        self.draw(data)
        # 3. 预处理图片
        train_dataset = self.transformTrainImages()
        test_dataset = self.transformTestImages()
        # 4. 加载DataLoader
        train_loader = self.getDataLoader(train_dataset)
        test_loader = self.getDataLoader(test_dataset)
        # 5. 将DataLoader存入文件
        self.object2file(train_loader,r'./static/object/train_loader.pkl')
        self.object2file(test_loader,r'./static/object/test_loader.pkl')
        return data

    # 把对象写入文件
    def object2file(self,object,path):
        with open(path,'wb') as f:
            pickle.dump(object,f)
        f.close()

    # 可视化生成图片保存
    def draw(self,data):
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        x, train, test = [], [], []
        for element in data:
            x.append(element[0])
            train.append(element[1])
            test.append(element[2])
        fig = plt.figure()
        plt.bar(x, train, label='train')
        plt.bar(x, test, bottom=train, label='test')
        plt.xlabel('类别')
        plt.ylabel('图片数量')
        plt.legend()
        plt.savefig(r'./static/prepare/visual.png')
        plt.close(fig)

    # 输入路径，返回该路径下的所有文件夹名称（这些名称就是类别名称）
    def getFolderName(self):
        folders = []
        for foldername in os.listdir(self.input_path):
            if os.path.isdir(os.path.join(self.input_path, foldername)):
                folders.append(foldername)
        return folders

    # 将原数据随机分布到训练集和测试集
    def divideImages(self):
        # 获取原数据类别
        cates = self.getFolderName()
        data = []
        # 创建train和test文件夹
        self.train_path = self.output_path + '\\train'
        self.test_path = self.output_path + '\\test'
        os.mkdir(os.path.join(self.train_path))
        os.mkdir(os.path.join(self.test_path))
        # 在train和test文件夹分布创建各类别的文件夹
        for cate in cates:
            os.mkdir(os.path.join(self.train_path, cate))
            os.mkdir(os.path.join(self.test_path, cate))
            # 训练集和测试集路径
            tmp_train_path = self.train_path + '\%s' % cate
            tmp_test_path = self.test_path + '\%s' % cate
            # 获取所有图片的路径
            image_paths = [os.path.join(self.input_path + '\%s' % cate, image) for image in
                           os.listdir(self.input_path + '\%s' % cate) if
                           image.endswith('.jpg') or image.endswith('.png') or image.endswith('jpeg')]
            # 打乱图片路径
            random.shuffle(image_paths)
            # 训练集：测试集=7:3
            split_index = int(len(image_paths) * 0.7)
            # 分割图片路径
            train_image_paths = image_paths[:split_index]
            test_image_paths = image_paths[split_index:]
            data.append([cate, len(train_image_paths), len(test_image_paths), len(image_paths)])
            # 移动图片到训练集目录
            for train_image_path in train_image_paths:
                shutil.copy(train_image_path, os.path.join(tmp_train_path, os.path.basename(train_image_path)))
            # 移动图片到测试集目录
            for test_image_path in test_image_paths:
                shutil.copy(test_image_path, os.path.join(tmp_test_path, os.path.basename(test_image_path)))
        return cates,data

    # 对训练集图像预处理：缩放裁剪，图像增强，转Tensor，归一化
    def transformTrainImages(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # 载入训练集
        train_dataset = datasets.ImageFolder(self.train_path, train_transform)
        # tmp_paths = [train_dataset.imgs[0],train_dataset.imgs[-1]]
        # for i,element in enumerate(tmp_paths):
        #     tmp_path = element[0]
        #     shutil.copy(tmp_path,'./static/prepare/train_%s.jpg'%(i+1))
        #     image = Image.open(tmp_path)
        #     processed_image = train_transform(image)
        #     self.saveRandomImages(processed_image,'train_processed_%s'%(i+1))
        self.saveNPY(train_dataset)
        return train_dataset

    # 对测试集图像预处理：缩放，裁剪，转Tensor，归一化
    def transformTestImages(self):
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # 载入测试集
        test_dataset = datasets.ImageFolder(self.test_path, test_transform)
        # tmp_paths = [test_dataset.imgs[0],test_dataset.imgs[-1]]
        # for i,element in enumerate(tmp_paths):
        #     tmp_path = element[0]
        #     shutil.copy(tmp_path, './static/prepare/test_%s.jpg'%(i+1))
        #     image = Image.open(tmp_path)
        #     processed_image = test_transform(image)
        #     self.saveRandomImages(processed_image,'test_processed_%s'%(i+1))
        return test_dataset

    # 保存索引到类别，类别到索引的映射关系到npy文件
    def saveNPY(self,dataset):
        idx_to_labels = {y:x for x,y in dataset.class_to_idx.items()}
        np.save('./static/prepare/idx_to_labels.npy',idx_to_labels)
        np.save('./static/prepare/labels_to_idx.npy',dataset.class_to_idx)

    def getDataLoader(self,dataset):
        data_loader = DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4)
        images,labels = next(iter(data_loader))
        self.saveRandomImages(images[0],'batch_%s'%self.i,labels[self.i-1].item())
        self.saveOriginImages(images[0],'origin_%s'%self.i,labels[self.i-1].item())
        self.i += 1
        return data_loader

    def saveOriginImages(self,image,image_name,label):
        fig = plt.figure()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        plt.imshow(np.clip(image.permute(*[1,2,0])*std+mean, 0, 1))
        plt.title('label:' + str(label))
        plt.savefig('./static/prepare/%s.jpg'%image_name)
        plt.close(fig)

    def saveRandomImages(self,image,image_name,label):
        fig = plt.figure()
        plt.imshow(image.permute((1,2,0)))
        plt.title('label:'+str(label))
        plt.savefig('./static/prepare/%s.jpg'%image_name)
        plt.close(fig)
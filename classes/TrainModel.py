import pickle
from tqdm import tqdm
import torch
from torchvision import models
import torch.optim as optim
import torch.nn as nn

class TrainModel:
    def __init__(self, epochs, option):
        self.model = None
        self.epochs = epochs
        self.option = option
        self.device = 'cpu'
        self.optimizer = None

    def startTraining(self):
        # 加载train_loader,test_loader
        train_loader, test_loader = self.file2Object()
        # 根据option，选择模型，optimizer
        if self.option == 'option1':
            self.trainOption1()
        elif self.option == 'option2':
            self.trainOption2()
        elif self.option == 'option3':
            self.trainOption3()
        # 模型加载至cpu，返回交叉熵损失函数
        criterion = self.trainSetting()
        # 循环训练
        print('option=%s,epochs=%s'%(self.option,self.epochs))
        print(self.optimizer)
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            for images, labels in train_loader:  # 获取训练集的一个batch，包含数据和标注
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)  # 前向预测，获得当前batch的预测结果
                loss = criterion(outputs, labels)  # 比较预测结果和标注，计算当前batch的交叉熵损失函数

                self.optimizer.zero_grad()
                loss.backward()  # 损失函数对神经网络权重反向传播求梯度
                self.optimizer.step()  # 优化更新神经网络权重

        self.testOntestDataset(test_loader)
        # 保存模型
        torch.save(self.model,'./static/model/model_%s.pth'%self.option[-1])

    def file2Object(self):
        with open('./static/object/train_loader.pkl', 'rb') as f:
            train_loader = pickle.load(f)
        f.close()
        with open('./static/object/test_loader.pkl', 'rb') as f:
            test_loader = pickle.load(f)
        f.close()
        return train_loader, test_loader

    def trainOption1(self):
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.optimizer = optim.Adam(self.model.fc.parameters())

    def trainOption2(self):
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.optimizer = optim.Adam(self.model.parameters())

    def trainOption3(self):
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.optimizer = optim.Adam(self.model.parameters())

    def trainSetting(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        # 交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        return criterion

    # 在测试集上初步测试
    def testOntestDataset(self,test_loader):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images,labels in tqdm(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _,preds = torch.max(outputs,1)
                total += labels.size(0)
                correct += (preds == labels).sum()

            print('correct:%s,total:%s'%(correct,total))
            accuracy = 100*correct/total
            print('测试集上的准确率为{:.3f}%'.format(accuracy))
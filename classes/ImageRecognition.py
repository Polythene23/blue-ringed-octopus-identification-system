import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image, ImageDraw

class ImageRecognition:

    def __init__(self,option,image_path,new_model_path=''):
        self.model = None
        self.option = option
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.new_model_path = new_model_path
        self.image_path = image_path

    def start(self):
        self.importModel()
        idx_to_labels,labels_to_idx = self.importNPY()
        img_pil,pred_softmax = self.transformImage()
        result_data = self.draw(idx_to_labels,img_pil,pred_softmax)
        return result_data

    # 导入npy文件
    def importNPY(self):
        idx_to_labels = np.load('./static/prepare/idx_to_labels.npy', allow_pickle=True).item()
        labels_to_idx = np.load('./static/prepare/labels_to_idx.npy',allow_pickle=True).item()
        return idx_to_labels,labels_to_idx

    # 导入模型
    def importModel(self):
        if self.option == 'option1':
            self.model = torch.load('./static/model/model_1.pth')
        elif self.option == 'option2':
            self.model = torch.load('./static/model/model_2.pth')
        elif self.option == 'option3':
            self.model = torch.load('./static/model/model_3.pth')
        elif self.option == 'option4':
            self.model = torch.load(self.new_model_path)
        self.model = self.model.eval().to(self.device)

    # 图片预处理,前向预测
    def transformImage(self):
        img_pil = Image.open(self.image_path)
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])
        input_img = test_transform(img_pil)
        input_img = input_img.unsqueeze(0).to(self.device)
        # 执行前向预测，得到所有类别的logit预测分数
        pred_logits = self.model(input_img)
        # 对logit分数做softmax运算
        pred_softmax = F.softmax(pred_logits,dim=1)
        return img_pil,pred_softmax

    def draw(self,idx_to_labels,img_pil,pred_softmax):
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        fig = plt.figure(figsize=(18,6))

        # 绘制左图
        ax1 = plt.subplot(1,2,1)
        ax1.imshow(img_pil)
        ax1.axis('off')

        # 绘制右图
        ax2 = plt.subplot(1,2,2)
        x = idx_to_labels.values()
        y = pred_softmax.cpu().detach().numpy()[0] * 100
        ax = plt.bar(x, y, width=0.3)
        ax2.bar(x,y,alpha=0.5,width=0.3,color='blue',edgecolor='yellow',lw=2,align='center')
        plt.bar_label(ax,fmt='%.2f',fontsize=25)# 置信度数值
        plt.title('图像分类预测结果',fontsize=30)
        plt.xlabel('类别',fontsize=25)
        plt.ylabel('置信度',fontsize=25)
        plt.ylim([0,110])
        ax2.tick_params(labelsize=16)
        plt.xticks(rotation=0)

        plt.tight_layout()
        fig.savefig(r'./static/result/result.jpg')
        plt.close(fig)

        n = 2
        top_n = torch.topk(pred_softmax,n)# 取置信度最大的n个结果
        pred_ids = top_n[1].cpu().detach().numpy().squeeze()# 解析出类别
        confs = top_n[0].cpu().detach().numpy().squeeze()# 解析出置信度

        # 获取表格数据
        result_data = []
        for i in range(2):
            id = int(pred_ids[i])  # 添加类别号
            name = idx_to_labels[pred_ids[i]]# 添加类别名称
            confidence = round(confs[i]*100,2)# 添加置信度
            result_data.append([id,name,confidence])
        return result_data
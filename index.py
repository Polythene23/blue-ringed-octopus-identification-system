#coding=utf-8
import os
from classes.Downloader import Downloader
from classes.Prepare import Prepare
from classes.TrainModel import TrainModel
from classes.ImageRecognition import ImageRecognition
import web
import pymysql
import hashlib
import tempfile
import json
import matplotlib.pyplot as plt
import time

def sqlSelect(sql):
    conn=pymysql.connect(host='localhost',port=3306,user='root',passwd='123456',db='web')
    cur = conn.cursor()
    cur.execute(sql)
    sqlData=cur.fetchall()
    cur.close()
    conn.close()
    return sqlData
def sqlWrite(sql):
    conn=pymysql.connect(host='localhost',port=3306,user='root',passwd='123456',db='web')
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()
    conn.commit()
    conn.close()
    return

urls = (
    '/login.html','login',
    '/index.html', 'index',
    '/download.html','Download',
    '/preparedata.html','PrepareData',
    '/trainmodel.html','TrainModel',
    '/recognition.html','Recognition',
)

class index:
    def GET(self):
        return render.index()
    def POST(self):
        return render.index()

class Download:

    def GET(self):
        return render.download('','','','','','')

    def POST(self):
        webData = web.input()
        if 'submitBtn1' in webData:
            pic_keyword = webData.get('pic_keyword')
            search_engine = webData.get('search_engine')
            download_addr = webData.get('download_addr')
            pic_name = webData.get('pic_name')
            pic_num = webData.get('pic_num')
            # 创建下载图片类Download的对象myDownloader
            myDownloader = Downloader(pic_keyword, pic_name, int(pic_num), download_addr, search_engine)
            # 获取所有图片地址，仅供预览
            pic_urls = myDownloader.getPicUrls()
        else:
            data = json.loads(web.data())
            pic_keyword = data['params'][0]['p_k']
            search_engine = data['params'][0]['s_e']
            download_addr = data['params'][0]['d_a']
            pic_name = data['params'][0]['p_name']
            pic_num = data['params'][0]['p_num']
            myDownloader = Downloader(pic_keyword, pic_name, int(pic_num), download_addr,search_engine)
            pic_urls = [element['url'] for element in data['images']]
            myDownloader.save(pic_urls)

        return render.download(pic_keyword, search_engine, download_addr, pic_name, pic_num, pic_urls)

class PrepareData:

    def GET(self):
        return render.preparedata([['','','','']])

    def POST(self):
        data = web.input()
        input_path = data.get('input_path')
        output_path = data.get('output_path')
        p = Prepare(input_path,output_path)
        visual_data = p.start()
        return render.preparedata(visual_data)

class TrainModel:
    def GET(self):
        return render.trainmodel()

    def POST(self):
        data = json.loads(web.data())
        epochs = data['params'][0]['epoch']
        option = data['params'][0]['train_option']
        tm = TrainModel(int(epochs),option)
        tm.startTraining()
        return render.trainmodel()

class Recognition:
    option = 'option1'
    new_model_path = ''

    def GET(self):
        return render.recognition([['','','']])

    def POST(self):
        data = web.input()
        result_data = [['','','']]
        if 'submitBtn' in data:
            # 保存用户上传的图片
            timestamp = int(time.time())
            file_path = f"./static/userUpload/{timestamp}.png"
            with open(file_path,'wb') as f:
                f.write(data['imageInput'])
            f.close()
            # 识别图像
            ir = ImageRecognition(self.option,file_path)
            result_data = ir.start()
        elif 'saveBtn' in data:
            self.option = data['optionsRadios']
        # 用户选中自定义模型
        if self.option == 'option4':
            timestamp = int(time.time())
            file_path = f"./static/model/{timestamp}.pth"
            with open(file_path,'wb') as f:
                f.write(data['userModel'])
            f.close()
            self.new_model_path = file_path
            ir = ImageRecognition(self.option,file_path,new_model_path=self.new_model_path)
            result_data = ir.start()

        return render.recognition(result_data)

render = web.template.render('templates/')
web.config.debug = False
app = web.application(urls, globals())
root = tempfile.mkdtemp()
store = web.session.DiskStore(root)
session = web.session.Session(app, store)
if __name__ == "__main__":
    app.run()

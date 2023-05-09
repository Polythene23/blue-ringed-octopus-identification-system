import re
import requests
from pyquery import PyQuery as pq


class Downloader:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'
    }

    def __init__(self, keyword, pic_name, pic_num, path, search_engine):
        self.keyword = keyword
        self.pic_num = pic_num
        self.pic_name = pic_name
        self.path = path
        self.search_engine = search_engine

    # 获取所有要下载的图片的地址
    def getPicUrls(self):
        if self.search_engine == '百度':
            # 获取所有图片地址
            pic_urls = self.getBaiduPicUrls()
        elif self.search_engine == '必应':
            pic_urls = self.getBingPicUrls()
        return pic_urls

    # 先调用方法获取所有要下载图片的地址，再调用save方法将这些图片保存
    def start(self):
        pic_urls = self.getPicUrls()
        # 保存所有图片
        self.save(pic_urls)

    # 输入要下载的图片数量，计算并返回下载次数（一次可下载30张图片）
    def caculateTimes(self, pic_num):
        times = 1
        while 1:
            pic_num = pic_num - 30
            if pic_num > 0:
                times += 1
            else:
                break
        return times

    # 获取所有百度图片地址
    def getBaiduPicUrls(self):
        times = self.caculateTimes(self.pic_num)
        pic_urls = self.getBaiduPicUrlsFirst()
        times -= 1
        pn, gsm = 30, '1e'
        while True:
            if times > 0:
                tmp_urls, pn, gsm = self.getBaiduPicUrlsSecond(pn, gsm)
                pic_urls.extend(tmp_urls)
                times -= 1
            else:
                break
        return pic_urls[:self.pic_num]

    # 获取百度前30张图片地址
    def getBaiduPicUrlsFirst(self):
        # 先获取前30张图片url
        url = 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&sf=1&fmq=&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&fm=index&pos=history&word=%s' % self.keyword
        params = {
            'tn': 'baiduimage',
            'ipn': 'r',
            'ct': '201326592',
            'cl': '2',
            'lm': '-1',
            'st': '-1',
            'sf': '1',
            'fmq': '',
            'pv': '',
            'ic': '0',
            'nc': '1',
            'z': '',
            'se': '1',
            'showtab': '0',
            'fb': '0',
            'width': '',
            'height': '',
            'face': '0',
            'istype': '2',
            'ie': 'utf-8',
            'fm': 'index',
            'pos': 'history',
            'word': self.keyword
        }
        html = requests.get(url, headers=self.headers, params=params)
        html = html.text
        pic_urls = re.findall('"thumbURL":"(.*?)",', html, re.S)
        return pic_urls

    # 继续获取其他百度图片地址
    def getBaiduPicUrlsSecond(self, pn, gsm):
        url = 'https://image.baidu.com/search/index'
        params = {
            'tn': 'resultjson_com',
            'ipn': 'rj',
            'ct': '201326592',
            'is': '',
            'fp': 'result',
            'fr': '',
            'word': self.keyword,
            'queryWord': self.keyword,
            'cl': '2',
            'lm': '-1',
            'ie': 'utf-8',
            'oe': 'utf-8',
            'adpicid': '',
            'st': '-1',
            'z': '',
            'ic': '0',
            'hd': '',
            'latest': '',
            'copyright': '',
            's': '',
            'se': '',
            'tab': '',
            'width': '',
            'height': '',
            'face': '0',
            'istype': '2',
            'qc': '',
            'nc': '1',
            'expermode': '',
            'nojc': '',
            'isAsync': '',
            'pn': pn,
            'rn': '30',
            'gsm': gsm
        }
        html = requests.get(url, headers=self.headers, params=params)
        raw_data = html.json()
        gsm = raw_data['gsm']
        pn = int(pn) + 30
        tmp_urls = []
        for element in raw_data['data'][:-1]:
            tmp_urls.append(element['hoverURL'])

        return tmp_urls, pn, gsm

    # 获取必应所有图片地址，功能：根据pic_num控制翻页，extend所有图片地址
    def getBingPicUrls(self):
        pic_count = 0
        sfx = 0  # 必应的一个翻页参数
        pic_urls = []
        while True:
            if pic_count < self.pic_num:
                tmp_urls = self.getBingPicUrlsInAction(sfx)
                pic_urls.extend(tmp_urls)
                pic_count = len(pic_urls)
                sfx += 1
            else:
                break
        return pic_urls[:self.pic_num]

    # 根据页码参数获取相应必应网页的图片地址
    def getBingPicUrlsInAction(self, SFX):
        url = 'https://cn.bing.com/images/search?q=%s&SFX=%s' % (self.keyword, SFX)
        html = requests.get(url)
        data = pq(html.text)
        pic_urls = []
        for element in data('.imgpt img').items():
            pic_url = element.attr('data-src' if element.attr('src') == None else 'src')
            pic_urls.append(pic_url)
        return pic_urls

    # 输入图片地址的列表，将图片下载到本地
    def save(self, pic_urls):
        for i, pic_url in enumerate(pic_urls):
            pic = requests.get(pic_url)
            file_name = f"%s\%s{i + 1}.jpg" % (self.path, self.pic_name)
            print('正在下载第%s张图片：%s' % (i + 1, file_name))
            f = open(file_name, 'wb')
            f.write(pic.content)
        f.close()

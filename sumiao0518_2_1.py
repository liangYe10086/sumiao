# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:19:22 2018

@author: wangzhaoliang
"""

from PIL import Image
import numpy as np
import logging
import os
import json
#import 
import urllib
from PIL import Image
from io import BytesIO
import cv2
import sys,time
import urllib
import requests
from time import time
#import NosUtils
#import tools
#import uuid
#import tools
#import tools.nosUtils
#from tools.nosUtils import NosUtils
#from tools import Properties

logger = logging.getLogger("Sumiao")

class Main():
    def __init__(self):
        pass
    
    def predict(self,**kwargs):
#        print(kwargs)
        if 'params' in kwargs:
            param = kwargs['params']
#            print(param)
            if (not self.check_json(param)):
                print('aaa')
                print(type(param))
                logger.info(
                    "[sumiao.predict] format of json param is error ! param : %s", kwargs)
                return 'param error'
            info = json.loads(param)
            param_path = info['image_path']
            tu_type_ = info['tu_type']

            dir = self.total(url=param_path,tu_type=tu_type_)

            flag = os.path.exists(dir)

#            if (flag):
#                nosKey = self.uploadToNos(dir)
#                urlPrefix = Properties.get('nos_url', 'nos')
#                os.remove(dir)
#                return '%s%s' % (urlPrefix, nosKey)
#            else:
#                return "failed"

        else:   
            logger.info('Must have an input as path param ')
            return 'param error'

    def sumiao(self,url, number = 5):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        width  = image.size[0]
        height = image.size[1]
        
        if width > height:
            width = int(width/(height/256))
            height = 256
            image = image.resize((width, height))
        else:
            height = int(height/(width/256))
            width = 256
            image = image.resize((width, height))
        in_path='./input/'+url.split('/')[-1]
        image.save(in_path)
        number=number
        print('Transferring '+url.split('/')[-1])
        end = './output/' + url.split('/')[-1].replace('.jpg','')+'_sumiao.jpg' 
        a = np.asarray(image.convert('L')).astype('float')
        depth = number  # (0-100)
        grad = np.gradient(a)  # 取图像灰度的梯度值
        grad_x, grad_y = grad  # 分别取横纵图像梯度值
        grad_x = grad_x * depth / 100.
        grad_y = grad_y * depth / 100.
        A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)
        uni_x = grad_x / A
        uni_y = grad_y / A
        uni_z = 1. / A
        vec_el = np.pi / 2.2  # 光源的俯视角度，弧度值
        vec_az = np.pi / 4.  # 光源的方位角度，弧度值
        dx = np.cos(vec_el) * np.cos(vec_az)  # 光源对x 轴的影响
        dy = np.cos(vec_el) * np.sin(vec_az)  # 光源对y 轴的影响
        dz = np.sin(vec_el)  # 光源对z 轴的影响
        b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)  # 光源归一化
        b = b.clip(0, 255)
        im = Image.fromarray(b.astype('uint8'))  # 重构图像
        im.save(end)
        os.remove(in_path)
        return end
    
    def fudiao(self,url):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        width  = img.size[0]
        height = img.size[1]
        
        if width > height:
            width = int(width/(height/256))
            height = 256
            img = img.resize((width, height))
        else:
            height = int(height/(width/256))
            width = 256
            img = img.resize((width, height))
        in_path='./input/'+url.split('/')[-1]
        img.save(in_path)
        print('Transferring '+url.split('/')[-1])
        end = './output/' + url.split('/')[-1].replace('.jpg','')+'_fudiao.jpg' 
        
        img_array=img.load()
        width=img.size[0]
        hight=img.size[1]
        mode='RGB'
        d=128
        ret=Image.new(mode, (width,hight), color=0)
        for i in range(width-1):
            for j in range(hight-1):
                pix1=img.getpixel((i,j))
                pix2=img.getpixel((i+1,j+1))
                aver0=(pix1[0]-pix2[0]+d) > 255
                if aver0  :
                    r=255
                else:
                    r=pix1[0]-pix2[0]+d
                aver1=(pix1[1]-pix2[1]+d) > 255
                if aver1  :
                    g=255
                else:
                    g=pix1[1]-pix2[1]+d
                aver2=(pix1[2]-pix2[2]+d) > 255
                if aver2  :
                    b=255
                else:
                    b=pix1[2]-pix2[2]+d
                if r < 0 :
                    r = 0
                if g < 0 :
                    g = 0
                if b < 0 :
                    b = 0
                ret.putpixel((i, j), (r,g,b))
        ret.save(end)
        os.remove(in_path)
        return end
    
    def muke(self,url):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        width  = img.size[0]
        height = img.size[1]
        
        if width > height:
            width = int(width/(height/256))
            height = 256
            img = img.resize((width, height))
        else:
            height = int(height/(width/256))
            width = 256
            img = img.resize((width, height))
        
        in_path='./input/'+url.split('/')[-1]
        img.save(in_path)
        print('Transferring '+url.split('/')[-1])
        end = './output/' + url.split('/')[-1].replace('.jpg','')+'_muke.jpg' 
        
        
        img_array=img.load()
        width=img.size[0]
        hight=img.size[1]
        step = 100.0/width
        mode='RGB'
        ret=Image.new(mode, (width,hight), color=0)
        for i in range(width):
            for j in range(hight):
                pix=img.getpixel((i,j))
                aver=(pix[0]+pix[1]+pix[2])/3
                if aver > 128 :
                    ret.putpixel((i, j), (255,255,255)) #白背景
                else:
                    ret.putpixel((i, j), (0,0,0))
        ret.save(end)
        os.remove(in_path)
        return end
    
    def maoboli(self,url):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        
        width  = img.size[0]
        height = img.size[1]
        
        if width > height:
            width = int(width/(height/256))
            height = 256
            img = img.resize((width, height))
        else:
            height = int(height/(width/256))
            width = 256
            img = img.resize((width, height))
            
          
    
       
        in_path='./input/'+url.split('/')[-1]
        img.save(in_path)
        print('Transferring '+url.split('/')[-1])
        end = './output/' + url.split('/')[-1].replace('.jpg','')+'_maoboli.jpg' 
        img_array=img.load()
        width=img.size[0]
        hight=img.size[1]
        step = 100.0/width
        #
        mode='RGB'
        r = 5
        ret=Image.new(mode, (width,hight), color=0)
        ret1=ret
        for i in range(width):
            for j in range(hight):
                a=i+np.random.randint(0, high=2*r)-r
                b=j+np.random.randint(0, high=2*r)-r
                if a >= width:
                    a=width-1
                if a < 0:
                    a=0
                if b >= hight:
                    b=hight-1
                if b < 0:
                    b=0
                pix=img.getpixel((a,b))
                ret.putpixel((i,j), pix)
        ret.save(end)
        os.remove(in_path)
        return end
    
    def katong(self,url):
        resp = urllib.urlopen(url)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
    #    color_image = img
        height = img.shape[0]
        width  = img.shape[1]
        
        if width > height:
            width = int(width/(height/256))
            height = 256
            img = cv2.resize(img, (width,height), interpolation=cv2.INTER_AREA)
        else:
            height = int(height/(width/256))
            width = 256
            img = cv2.resize(img, (width,height), interpolation=cv2.INTER_AREA)
        
        in_path='./input/'+url.split('/')[-1]
        cv2.imwrite(in_path,img)
    #    img.save(in_path)
        print('Transferring '+url.split('/')[-1])
        end = './output/' + url.split('/')[-1].replace('.jpg','')+'_katong.jpg' 
        
        num_down = 2  # 缩减像素采样的数目
        num_bilateral = 7 # 定义双边滤波的数目
    #    img_rgb = cv2.imread(r'C:\Users\wangzhaoliang\Desktop\photo_face\55.jpg')
        img_rgb = img
    
        # 用高斯金字塔降低取样
        img_color = img_rgb
        for _ in range(num_down):
            img_color = cv2.pyrDown(img_color)
        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=9,
                 sigmaColor=9,
                 sigmaSpace=7)
        # 升采样图片到原始大小
        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)
          
        # 转换为灰度并使其产生中等的模糊
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                 cv2.ADAPTIVE_THRESH_MEAN_C,
                 cv2.THRESH_BINARY,
                 blockSize=9,
                 C=6)
        # 转换回彩色图像
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        min_shape0=min(img_edge.shape[0],img_color.shape[0])
        min_shape1=min(img_edge.shape[1],img_color.shape[1])
        img_color1=img_color[:min_shape0,:min_shape1]
        img_edge1=img_edge[:min_shape0,:min_shape1]
        img_cartoon = cv2.bitwise_and(img_color1, img_edge1)
        cv2.imwrite(end,img_cartoon)
        os.remove(in_path)
        return end
    
    def youhua(self,url):
        resp = urllib.urlopen(url)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        color_image = img
        height = color_image.shape[0]
        width  = color_image.shape[1]
        
        if width > height:
            width = int(width/(height/256))
            height = 256
            color_image = cv2.resize(color_image, (width,height), interpolation=cv2.INTER_AREA)
        else:
            height = int(height/(width/256))
            width = 256
            color_image = cv2.resize(color_image, (width,height), interpolation=cv2.INTER_AREA)
            
        in_path='./input/'+url.split('/')[-1]
        cv2.imwrite(in_path,img)
        print('Transferring '+url.split('/')[-1])
        end = './output/' + url.split('/')[-1].replace('.jpg','')+'_youhua.jpg' 
        height = color_image.shape[0]
        width  = color_image.shape[1]
        channel = color_image.shape[2]
        
        if width > height:
            width = int(width/(height/256))
            height = 256
            color_image = cv2.resize(color_image, (width,height), interpolation=cv2.INTER_AREA)
        else:
            height = int(height/(width/256))
            width = 256
            color_image = cv2.resize(color_image, (width,height), interpolation=cv2.INTER_AREA)    
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        oil_image = np.zeros((height, width, channel))
        
        R = 3
        stroke_mask = []
        for y in range(-R, R):
            for x in range(-R, R):
                if y*y + x*x < R*R:
                    stroke_mask.append( (y,x) )
        
        for y in range(height):
            for x in range(width):
                local_histogram = np.zeros(256)
                local_channel_count = np.zeros((channel, 256))
                for dy,dx in stroke_mask:
                    yy = y+dy
                    xx = x+dx
                    if yy < 0  or yy >= height or xx <= 0  or xx >= width:
                        continue
                    intensity = gray_image[yy, xx]
                    local_histogram[intensity] += 1
                    for c in range(channel):
                        local_channel_count[c, intensity] += color_image[yy, xx, c]
        
                max_intensity = np.argmax(local_histogram)
                max_intensity_count = local_histogram[max_intensity]
                for c in range(channel):
                    oil_image[y,x,c] = local_channel_count[c, max_intensity] / max_intensity_count
    
        oil_image = oil_image.astype('int')
        cv2.imwrite(end, oil_image)
        os.remove(in_path)
        return end
    
    
    def total(self,url,tu_type):
        if not os.path.exists('./input/'):
            os.mkdir('./input/')    
        if not os.path.exists('./output/'):
            os.mkdir('./output/')      
        if tu_type == 'sumiao':
            return self.sumiao(url, number = 5)
        elif tu_type == 'youhua':
            return self.youhua(url)
        elif tu_type == 'fudiao':
            return self.fudiao(url)
        elif tu_type == 'muke':
            return self.muke(url)
        elif tu_type == 'maoboli':
            return self.maoboli(url)
        elif tu_type == 'katong':
            return self.katong(url)  
        else:
#            print('type not found')
            return None
        
    def uploadToNos(self, fileName):
        target = 'zww/image/%s%s' % (uuid.uuid4(), os.path.splitext(fileName)[1])
        nosUtils = NosUtils()
        nosUtils.upload(target, fileName)
        return target

    def check_json(self, jsonInfo):
        ret = True
        try:
            json.loads(jsonInfo)
        except ValueError:
            ret = False
        return ret
        



if __name__=='__main__':

    url_ = 'http://zongyipic.manmankan.com/yybpic/zongyi/201408/588_1011_4331.jpg'
    url = 'http://img.fangle.com/upfile/borough/picture/2013/10/15/11/20131015114731630.jpg'
    type1 = 'sumiao'
    type2 = 'youhua'
    type3 = 'katong'
    type4 = 'muke'
    type5 = 'fudiao'
    type6 = 'maoboli'
    ms = Main()
    param='{"image_path": "http://zongyipic.manmankan.com/yybpic/zongyi/201408/588_1011_4331.jpg","tu_type":"katong"}'
    ms.predict(params = param)


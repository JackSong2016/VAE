import os
import numpy as np
from PIL import Image

"""
    生成训练VAE模型所需要的90*90图片
    
"""
dataPath=r'F:\dataSet\disorder\newStructures'
def createImage(path):
    with open(path,'r') as file:
        context = [[float(j) for j in i.strip().split(" ")] for i in file.readlines()]
        for c in range(0, len(context)):
            for cc in range(0,len(context[c])):
                if context[c][cc] == 1.0:
                    context[c][cc] = 0.0
                else:
                    context[c][cc] = 255.0
    file.close()
    with open(path,'r') as fi:
        # 读取文件二进制串并将其转换成字符串
        binIndex="".join(["".join('%s' % i for i in item) for item in [[int(j) for j in j.strip().split(" ")] for j in fi.readlines()]])
        decimalIndex=str(int(binIndex,2))
    fi.close()
    data = np.array(context)
    data = data.repeat(6, axis=0)
    data = data.repeat(6, axis=1)
    new_map = Image.fromarray(data.astype('uint8'))     # 不加上unit8则保存图片会出现全黑
    if not os.path.exists("./Images20000"):
        os.mkdir("./Images20000")
    new_map.save('./Images20000/image_{}.png'.format(decimalIndex))
def getDataTensor():
    for inner in os.listdir(dataPath):
        featurePath=os.path.join(dataPath,inner,'feature.data')
        createImage(featurePath)
getDataTensor()

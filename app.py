import io
import os
import re
import base64
import urllib.error
import urllib.parse
import urllib.request
import uuid
from time import time

import pandas as pd
import numpy as np
from icecream import ic
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import transforms

from captcha_cnn_model import CNN
from captcha_setting import ALL_CHAR_SET, ALL_CHAR_SET_LEN

# 测试验证码 1748
image_data = "data:image/png;base64,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"

#  总署验证码接口
HC_URL = 'http://health.customsapp.com:18081/htdecl/rest/htdecl/getVerify?time=1620627535704&tel=18301530000'
new_HC_URL = 'https://health.customsapp.com/htdecl/rest/htdecl/getVerify?time=1620627535704&tel=18301530000'


# 20210702 总署验证码获取开始需要headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) '
                  'Version/14.1.1 Safari/605.1.15 '
}


"""
方案
识别失败时，后端程序保存识别失败的验证码base64以及用户手动输入的验证码至数据库；
当保存的验证码到达一定数量级时，（例如：1000个）
程序用这一千条数据训练出一个模型：【年月日-验证码数量】
将这1000个验证码保存至历史数据库
使用历史数据库保存的验证码重新训练模型：【年月日-验证码数量】
使用1000模型测试历史数据，查看准确率
使用【1000+历史】模型测试新获取数据，查看准确率
使用准确率更高的模型
两个模型的准确率都不理想时，增加验证码数量级
"""


def base64_to_tensor(data: str) -> str:  # 将base64编码转换为张量并升至4维
    try:
        base64_data = re.sub('^data:image/.+;base64,', '', data)
        byte_data = base64.b64decode(base64_data)
        image_data = io.BytesIO(byte_data)
        img = Image.open(image_data)
        img = img.convert('L')
        trans = transforms.ToTensor()(img)
        trans = torch.unsqueeze(trans, 1)  # 升维
        return trans
    except:
        return "ERR"


def predict(tensor_data):  # 预测张量图像
    try:
        cnn = CNN()
        cnn.eval()  # 为什么会提升精度
        cnn.load_state_dict(torch.load('model 20210702-1000.pkl'))
        v_image = Variable(tensor_data)
        predict_label = cnn(v_image)

        c0 = ALL_CHAR_SET[np.argmax(predict_label[0, 0:ALL_CHAR_SET_LEN].data.numpy())]
        c1 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN:2 * ALL_CHAR_SET_LEN].data.numpy())]
        c2 = ALL_CHAR_SET[np.argmax(predict_label[0, 2 * ALL_CHAR_SET_LEN:3 * ALL_CHAR_SET_LEN].data.numpy())]
        c3 = ALL_CHAR_SET[np.argmax(predict_label[0, 3 * ALL_CHAR_SET_LEN:4 * ALL_CHAR_SET_LEN].data.numpy())]
        # print(predict_label)
        c = '%s%s%s%s' % (c0, c1, c2, c3)
        # print(c)
        return c
    except:
        return "ERR"


def run(base64_data):  # 训练用例
    tensor_data = base64_to_tensor(base64_data)
    result = predict(tensor_data)
    return result




# 将base64转换成PIL图像
def test_for_image(data):
    base64_data = re.sub('^data:image/.+;base64,', '', data)
    byte_data = base64.b64decode(base64_data)
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data)
    return img


# 将PIL图像升1维并转换成张量
def image_tensor(image_data):
    # if not isinstance(image_data, PIL.JpegImagePlugin.JpegImageFile):
    #     return None
    trans = None
    try:
        img = image_data.convert('L')
        trans = transforms.ToTensor()(img)
        trans = torch.unsqueeze(trans, 1)  # 升维
    except Exception as e:
        print(e)
    finally:
        return trans


# 将base64图像数据或图像地址转换成PIL图像
def base64_or_file_2PIL(images_data):
    if re.match('^data:image/.+;base64,', images_data):
        base64_data = re.sub('^data:image/.+;base64,', '', images_data)
        byte_data = base64.b64decode(base64_data)
        bytes_image = io.BytesIO(byte_data)
        img = Image.open(bytes_image)
    else:
        img = Image.open(images_data)
    return img


# 打标用 识别并将图像重命名
def predict_the_data():
    #
    folder = '/Users/hao/Desktop/桌面 - Hao的MacBook Pro/发件箱/验证码识别/海关总署验证码验证集'
    data = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
    data.remove('/Users/hao/Desktop/桌面 - Hao的MacBook Pro/发件箱/验证码识别/海关总署验证码验证集/.DS_Store')
    for img in data:
        print(predict(image_tensor(base64_or_file_2PIL(img))))
        os.rename(img, img[:61] + predict(image_tensor(base64_or_file_2PIL(img))) + '_' + str(uuid.uuid4()) + '.png')


# 将csv中保存的识别失败的验证码识别并保存
def save_the_err():
    origin_data = pd.read_csv('/Users/hao/Desktop/桌面 - Hao的MacBook Pro/发件箱/验证码识别/'
                              'result1python_result2收费识别1000个.csv')
    img_datas = origin_data['base64resource'].to_list()
    for img in img_datas:
        base64_data = re.sub('^data:image/.+;base64,', '', img)
        byte_data = base64.b64decode(base64_data)
        bytes_image = io.BytesIO(byte_data)
        img_d = Image.open(bytes_image)
        cap = predict(image_tensor(base64_or_file_2PIL(img)))
        print(cap)
        img_d.save('/Users/hao/Desktop/桌面 - Hao的MacBook Pro/发件箱/验证码识别/异常集/'
                   + cap + '_' + str(uuid.uuid4()) + '.png')





# 获取验证码并转换成base64
def get_captcha(img_url):
    request = urllib.request.Request(img_url, headers=headers)
    try:
        response = urllib.request.urlopen(request)
        if response.getcode() == 200:
            base64_data = base64.b64encode(response.read())
            s = base64_data.decode()
            data = 'data:image/jpeg;base64,%s' % s
            return data
        else:
            print('ERR')
    except Exception as e:
        print(e)


# 测试用 下载新验证码并识别，重命名
def make_test(test_times: int = 10):
    for i in range(test_times):
        base64data = get_captcha(new_HC_URL)
        base64_data = re.sub('^data:image/.+;base64,', '', base64data)
        byte_data = base64.b64decode(base64_data)
        bytes_image = io.BytesIO(byte_data)
        img_d = Image.open(bytes_image)
        cap = predict(image_tensor(base64_or_file_2PIL(base64data)))
        print(cap, float(i / test_times))

        img_d.save('/Users/hao/Desktop/桌面 - Hao的MacBook Pro/发件箱/验证码识别/训练测试集/'
                   + cap + '_' + str(time()) + '.png')

# 测试用 显示一些base64格式验证码
def make_some_captcha(some: int):
    for i in range(some):
        base64data = get_captcha(new_HC_URL)
        print(base64data)


# 打标用 使用尖叫数据识别base64格式验证码
def JJSJ_verify(img_base64: str):
    host = 'http://apigateway.jianjiaoshuju.com'
    path = '/api/v_1/yzmCustomized.html'
    method = 'POST'
    appcode = '7E93771B49D7EE27BD8514472DF4B60E'
    appKey = 'AKID6935d0d2478a86483008682ac4bf6bce'
    appSecret = '55485f6c5f5aedea100dceda7a4f7362'
    bodys = {}
    url = host + path
    bodys['v_pic'] = img_base64
    bodys['pri_id'] = '''dn'''
    bodys['number'] = '''4'''
    headers['appcode'] = appcode
    headers['appKey'] = appKey
    headers['appSecret'] = appSecret
    data = urllib.parse.urlencode(bodys).encode('utf8')  # 对参数进行编码，解码使用 urllib.parse.urldecode
    result = 'ERR'

    try:
        request = urllib.request.Request(url, data, headers)  # 请求处理
        reponse = urllib.request.urlopen(request).read()
        if reponse:
            response_data = reponse.decode('utf8')
            response_data = eval(response_data)
            msg = response_data['msg']
            v_code = response_data['v_code']
            errCode = response_data['errCode']
            result = v_code
    except Exception as e:
        print(e)
    finally:
        return result


# 打标用 使用尖叫数据获取总署验证码识别并保存
def JJSJ_marking(some: int):
    for i in range(some):
        base64data = get_captcha(new_HC_URL)
        result = JJSJ_verify(base64data)
        base64_data = re.sub('^data:image/.+;base64,', '', base64data)
        byte_data = base64.b64decode(base64_data)
        bytes_image = io.BytesIO(byte_data)
        img_d = Image.open(bytes_image)
        print(result, i, "/", some)
        img_d.save('/Users/hao/Desktop/桌面 - Hao的MacBook Pro/发件箱/验证码识别/尖叫数据训练集复杂版/'
                   + result + '_' + str(time()) + '.png')


# 文件处理 批量修改文件名
def group_modify():
    path = r'/Users/hao/Desktop/桌面 - Hao的MacBook Pro/发件箱/验证码识别.nosync/尖叫数据训练集'
    i = 1
    # 对目录下的文件进行遍历
    for file in os.listdir(path):
        # 判断是否是文件
        if os.path.isfile(os.path.join(path, file)):
            # # 设置新文件名
            # new_name = file.replace(r'\w.p3g', r'\w.png')
            # # 重命名
            # os.rename(os.path.join(path, file), os.path.join(path, new_name))
            # print(file)

            portion = os.path.splitext(file)
            # 如果后缀是.txt
            if portion[1] == ".p3g":
                # 重新组合文件名和后缀名
                newname = portion[0] + ".png"
                os.rename(os.path.join(path, file), os.path.join(path, newname))


# 测试用 识别本地的验证码文件
def verify_local():
    folder = '/Users/hao/Desktop/桌面 - Hao的MacBook Pro/发件箱/验证码识别/旧版验证码'
    data = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
    # data.remove('/Users/hao/Desktop/桌面 - Hao的MacBook Pro/发件箱/验证码识别/旧版验证码/.DS_Store')
    for img in data:
        print(predict(image_tensor(base64_or_file_2PIL(img))))

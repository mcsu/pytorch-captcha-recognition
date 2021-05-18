import base64
import io
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import transforms
from captcha_cnn_model import CNN


from captcha_setting import ALL_CHAR_SET, ALL_CHAR_SET_LEN


def write_add(data):
    f = 'data.txt'
    with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        file.write(data + "nuctech")
        file.close()


def read_file():
    f = 'data.txt'
    with open(f) as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        content = file.readlines()
        file.close()

    lines = np.array(content)
    num_of_instances = lines.size
    for i in range(num_of_instances):
        img = lines[i].split("nuctech")  # ?
    for j in range(1000):
        print(run(img[j]))



def download_img(img_url, api_token):
    header = {"Authorization": "Bearer " + api_token}
    request = urllib.request.Request(img_url)

    basedir = os.path.abspath(os.path.dirname(__file__))
    path = basedir + os.path.sep + 'demo'

    try:
        response = urllib.request.urlopen(request)
        img_name = "data" + str(int(time.time())) + "img.txt"
        if response.getcode() == 200:
            # with open(img_name, "w") as f:
            #     print(path + os.path.sep + img_name)
            #     f.write(response.read())  # 将内容写入图片
            #     print(img_name)
            #     print('ok')

            base64_data = base64.b64encode(response.read())
            s = base64_data.decode()
            data = 'data:image/jpeg;base64,%s' % s
            write_add(data)

            return img_name
        else:
            print('ERR')

    except:
        return "failed"


def base64_to_tensor(data):  # 将base64编码转换为张量并升至4维

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
        cnn.load_state_dict(torch.load('model.pkl'))
        v_image = Variable(tensor_data)
        predict_label = cnn(v_image)

        c0 = ALL_CHAR_SET[np.argmax(predict_label[0, 0:ALL_CHAR_SET_LEN].data.numpy())]
        c1 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN:2 * ALL_CHAR_SET_LEN].data.numpy())]
        c2 = ALL_CHAR_SET[np.argmax(predict_label[0, 2 * ALL_CHAR_SET_LEN:3 * ALL_CHAR_SET_LEN].data.numpy())]
        c3 = ALL_CHAR_SET[np.argmax(predict_label[0, 3 * ALL_CHAR_SET_LEN:4 * ALL_CHAR_SET_LEN].data.numpy())]
        c = '%s%s%s%s' % (c0, c1, c2, c3)
        # print(c)
        return c
    except:
        return "ERR"


def run(base64_data):
    tensor_data = base64_to_tensor(base64_data)
    result = predict(tensor_data)
    return result


    base64_data = re.sub('^data:image/.+;base64,', '', data)
    byte_data = base64.b64decode(base64_data)
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data)
    img = img.convert('L')
    trans = transforms.ToTensor()(img)
    trans = torch.unsqueeze(trans, 1)  # 升维
    return trans


def predict(tensor_data):  # 预测张量图像
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl'))
    v_image = Variable(tensor_data)
    predict_label = cnn(v_image)

    c0 = ALL_CHAR_SET[np.argmax(predict_label[0, 0:ALL_CHAR_SET_LEN].data.numpy())]
    c1 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN:2 * ALL_CHAR_SET_LEN].data.numpy())]
    c2 = ALL_CHAR_SET[np.argmax(predict_label[0, 2 * ALL_CHAR_SET_LEN:3 * ALL_CHAR_SET_LEN].data.numpy())]
    c3 = ALL_CHAR_SET[np.argmax(predict_label[0, 3 * ALL_CHAR_SET_LEN:4 * ALL_CHAR_SET_LEN].data.numpy())]
    c = '%s%s%s%s' % (c0, c1, c2, c3)
    print(c)

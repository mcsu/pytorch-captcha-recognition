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

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALL_CHAR_SET_LEN = len(NUMBER)


def download_img(img_url, api_token):
    header = {"Authorization": "Bearer " + api_token}
    request = urllib.request.Request(img_url)

    try:
        response = urllib.request.urlopen(request)
        img_name = "data" + os.path.sep + str(int(time.time())) + "img.png"
        print(response.read())
        if response.getcode() == 200:
            with open(img_name, "wb") as f:
                f.write(response.read())  # 将内容写入图片

            return img_name
    except:
        return "failed"


def base64_to_tensor(data):  # 将base64编码转换为张量并升至4维
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

    c0 = NUMBER[np.argmax(predict_label[0, 0:ALL_CHAR_SET_LEN].data.numpy())]
    c1 = NUMBER[np.argmax(predict_label[0, ALL_CHAR_SET_LEN:2 * ALL_CHAR_SET_LEN].data.numpy())]
    c2 = NUMBER[np.argmax(predict_label[0, 2 * ALL_CHAR_SET_LEN:3 * ALL_CHAR_SET_LEN].data.numpy())]
    c3 = NUMBER[np.argmax(predict_label[0, 3 * ALL_CHAR_SET_LEN:4 * ALL_CHAR_SET_LEN].data.numpy())]
    c = '%s%s%s%s' % (c0, c1, c2, c3)
    print(c)


# str_raw = "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"
# tensor = base64_to_tensor(str_raw)
# predict(tensor)


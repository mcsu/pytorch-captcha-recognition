# -*- coding: UTF-8 -*-
import time

import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import transforms

import captcha_setting
import my_dataset
from captcha_cnn_model import CNN


def main():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model 20210702-1000.pkl'))

    predict_dataloader = my_dataset.get_predict_data_loader()

    for i, images in enumerate(predict_dataloader):
        v_image = Variable(images)
        print(type(v_image))
        print(v_image.size())

        predict_label = cnn(v_image)

        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c = '%s%s%s%s' % (c0, c1, c2, c3)
        print(c)


if __name__ == '__main__':
    main()


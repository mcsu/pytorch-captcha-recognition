import numpy


CHARACTER = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18,
             'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27,
             'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}

CAPTCHA_NUMBER = 6


def one_hot_encode(value: list) -> tuple:
    """编码，将字符转为独热码
    vector为独热码，order用于解码
    """
    order = []
    vector = numpy.zeros(CAPTCHA_NUMBER * len(CHARACTER ), dtype=float)
    for k, v in enumerate(value):
        index = k * len(CHARACTER) + CHARACTER.get(v)
        vector[index] = 1.0
        order.append(index)
    return vector


def one_hot_decode(value: list) -> str:
    """解码，将独热码转为字符
    """
    res = []
    for ik, iv in enumerate(value):
        val = iv - ik * len(CHARACTER) if ik else iv
        for k, v in CHARACTER.items():
            if val == int(v):
                res.append(k)
                break
    return "".join(res)


if __name__ == '__main__':
    code = '0A2JYD'
    vec = one_hot_encode(code)
    # print(orders)
    print(vec)
    # print(one_hot_decode(orders))

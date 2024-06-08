#通过爬虫从百度图片爬取七种飞机图片

import requests
import tqdm


def configs(search, page, number):
    """

    :param search:
    :param page:
    :param number:
    :return:
    """
    url = 'https://image.baidu.com/search/acjson'
    params = {
        "tn": "resultjson_com",
        "logid": "11555092689241190059",
        "ipn": "rj",
        "ct": "201326592",
        "is": "",
        "fp": "result",
        "queryWord": search,
        "cl": "2",
        "lm": "-1",
        "ie": "utf-8",
        "oe": "utf-8",
        "adpicid": "",
        "st": "-1",
        "z": "",
        "ic": "0",
        "hd": "",
        "latest": "",
        "copyright": "",
        "word": search,
        "s": "",
        "se": "",
        "tab": "",
        "width": "",
        "height": "",
        "face": "0",
        "istype": "2",
        "qc": "",
        "nc": "1",
        "fr": "",
        "expermode": "",
        "force": "",
        "pn": str(60 * page),
        "rn": number,
        "gsm": "1e",
        "1617626956685": ""
    }
    return url, params


def loadpic(number, page):
    """
    :param number:
    :param page:
    :return:
    """
    while (True):
        if number == 0:
            break
        url, params = configs(search, page, number)
        result = requests.get(url, headers=header, params=params).json()
        url_list = []
        for data in result['data'][:-1]:
            url_list.append(data['thumbURL'])
        for i in range(len(url_list)):
            getImg(url_list[i], 60 * page + i, path)#
            bar.update(1)
            number -= 1
            if number == 0:
                break
        page += 1
    print("\nfinish!")


def getImg(url, idx, path):
    """

    :param url:
    :param idx:
    :param path:
    :return:
    """
    img = requests.get(url, headers=header)
    file = open(path + "(" + str(idx + 1) + ")" '.jpg', 'wb')
    file.write(img.content)
    file.close()


number = ['歼20']
#需要搜索的关键词
for number_subset in number:
    if __name__ == '__main__':
        search = number_subset
        number = 800
        #需要爬取图片的数量
        path = 'D:/PYTHON/airplane/J-20/' + number_subset + 1
        #图片爬取后保存的文件夹位置
        header = {
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'
        }

        bar = tqdm.tqdm(total=number)
        page = 0
        loadpic(number, page)

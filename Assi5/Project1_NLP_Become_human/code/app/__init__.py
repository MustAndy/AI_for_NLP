from app import coreFunction as backend
from flask import Flask, redirect, url_for, request, jsonify,json,Blueprint,render_template,send_file
import flask
appStart = Flask(__name__)
from random import choice

import base64
@appStart.route('/')
def hello_world():
    return render_template('main.html')


@appStart.route('/success/<text>')
def success(text):
    """text = 中新网6月23日电 (记者潘旭临) 意大利航空首席商务官乔治先生22日在北京接受中新网记者专访时表示，意航确信中国市场对意航的重要性，目前意航已将发展中国市场提升到战略层级的高度，
               未来，意航将加大在华布局，提升业务水平。
               到意大利航空履职仅7个月的乔治，主要负责包括中国市场在内的亚太业务。
               乔治称，随着对华业务不断提升，意航明年可能会将每周4班提高到每天一班。同时，意航会借罗马新航站楼启用之际，吸引更多中国旅客到意大利旅游和转机。
               此外，还将加大对北京直飞航线的投资，如翻新航班座椅，增加电视中有关中国内容的娱乐节目、提高机上中文服务、餐饮服务、完善意航中文官方网站，提升商务舱和普通舱的舒适度等。"""

    result = backend.speech_extraction(text, say_word)
    #print(result)
    return 'welcome %s' % result[0][0]


@appStart.route('/jsondata', methods=['POST', 'GET'])
def infos():
    """
     请求的数据源，该函数模拟数据库中存储的数据，返回以下这种数据的列表：
    {'name': '香蕉', 'id': 1, 'price': '10'}
    {'name': '苹果', 'id': 2, 'price': '10'}
    """
    data = []
    names = ['香', '草', '瓜', '果', '桃', '梨', '莓', '橘', '蕉', '苹']
    for i in range(1, 1001):
        d = {}
        d['id'] = i
        d['name'] = choice(names) + choice(names)  # 随机选取汉字并拼接
        d['price'] = '10'
        data.append(d)
    if request.method == 'POST':
        print('post')
    if request.method == 'GET':
        info = request.values
        limit = info.get('limit', 10)  # 每页显示的条数
        offset = info.get('offset', 0)  # 分片数，(页码-1)*limit，它表示一段数据的起点
        print('get', limit)
        print('get  offset', offset)
        return jsonify({'total': len(data), 'rows': data[int(offset):(int(offset) + int(limit))]})
        # 注意total与rows是必须的两个参数，名字不能写错，total是数据的总长度，rows是每页要显示的数据,它是一个列表
        # 前端根本不需要指定total和rows这俩参数，他们已经封装在了bootstrap table里了
@appStart.errorhandler(404)
def not_found(e):
    return render_template('404.html')
import cv2
import numpy as np
from PIL import Image
@appStart.route('/result', methods=['POST', 'GET'])
def result():

    if request.method == 'POST':
        text = request.form['nm']
        #print(text)
        result,word_cloud = backend.speech_extraction(text, say_word)
        if result is not None:
            print(word_cloud)
            with open(word_cloud, 'rb') as img_f:
                img_stream = img_f.read()
                img_stream = base64.b64encode(img_stream)

            return render_template("result.html", result=result, img_stream=img_stream)
        else:
            return render_template('main.html')

    return 'success'


def start():
    global say_word
    say_word = backend.prepareModel()
    appStart.config['JSON_AS_ASCII'] = False
    #print(result)
    appStart.run()
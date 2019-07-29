from app import coreFunction as backend
from flask import Flask, redirect, url_for, request,json,Blueprint,render_template,send_file
import flask
appStart = Flask(__name__)


@appStart.route('/')
def hello_world():

    return send_file('./fontend/static/input.html')


@appStart.route('/success/<text>')
def success(text):
    """text = 中新网6月23日电 (记者潘旭临) 意大利航空首席商务官乔治先生22日在北京接受中新网记者专访时表示，意航确信中国市场对意航的重要性，目前意航已将发展中国市场提升到战略层级的高度，
               未来，意航将加大在华布局，提升业务水平。
               到意大利航空履职仅7个月的乔治，主要负责包括中国市场在内的亚太业务。
               乔治称，随着对华业务不断提升，意航明年可能会将每周4班提高到每天一班。同时，意航会借罗马新航站楼启用之际，吸引更多中国旅客到意大利旅游和转机。
               此外，还将加大对北京直飞航线的投资，如翻新航班座椅，增加电视中有关中国内容的娱乐节目、提高机上中文服务、餐饮服务、完善意航中文官方网站，提升商务舱和普通舱的舒适度等。"""

    result = backend.speech_extraction(text, say_word)
    print(result)
    return 'welcome %s' % result[0][0]


@appStart.route('/login/', methods=['POST', 'GET'])
def input():
    if request.method == 'POST':
        text = request.form['nm']
        print(text)
        return redirect(url_for('success', text=text))
    else:
        text = request.args.get('nm')

        print(text)
        return redirect(url_for('success', text=text))

    return 'success'


@appStart.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % username


@appStart.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id


if __name__ == '__main__':
    global say_word
    say_word = backend.prepareModel()
    appStart.config['JSON_AS_ASCII'] = False
    #print(result)
    appStart.run()
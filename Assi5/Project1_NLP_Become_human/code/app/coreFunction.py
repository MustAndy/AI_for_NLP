from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import re
import jieba
import os
from app import find_say_word, dependence_parser
from wordcloud import WordCloud

def cut(string): return ' '.join(jieba.cut(string))


ROOT_DIR = 'D:/senior/aiCourse/AI_for_NLP/Assi5/Project1_NLP_Become_human/code'
DATA_PATH = 'd:/senior/aiCourse/dataSource/'
MODEL_PATH = os.path.join(DATA_PATH, 'w2vmodel/wiki_news_cutting.model')
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'wiki_news_cutting.txt')
symbol = '[^(（|）|《|》|“|”|「|」|、|\-|：|·|；|?|,|.|:|"|\'|\')]'


def load_model():
    if os.path.exists(MODEL_PATH):
        print("Model found ! Now loading...")
        return Word2Vec.load(MODEL_PATH)
    else:
        print("Model not found, start training, please wait for about 15 minutes.")
        model = Word2Vec(LineSentence(open(TRAIN_DATA_PATH, encoding='gb18030')), size=100, min_count=1, workers=3)
        model.save(MODEL_PATH)
        return model


# 下面直接加载与训练模型


def prepareModel():
    all_word2vec = load_model()
    say_word = find_say_word.load_say_word(DATA_PATH, all_word2vec)
    return say_word


text3 = """中新网6月23日电 (记者潘旭临) 意大利航空首席商务官乔治先生22日在北京接受中新网记者专访时表示，意航确信中国市场对意航的重要性，目前意航已将发展中国市场提升到战略层级的高度，
未来，意航将加大在华布局，提升业务水平。
到意大利航空履职仅7个月的乔治，主要负责包括中国市场在内的亚太业务。
乔治称，随着对华业务不断提升，意航明年可能会将每周4班提高到每天一班。同时，意航会借罗马新航站楼启用之际，吸引更多中国旅客到意大利旅游和转机。
此外，还将加大对北京直飞航线的投资，如翻新航班座椅，增加电视中有关中国内容的娱乐节目、提高机上中文服务、餐饮服务、完善意航中文官方网站，提升商务舱和普通舱的舒适度等。"""


def token(string):
    return ''.join(re.findall(symbol, string))


def speech_extraction(text, say_word):
    words = dependence_parser.pyltp_cutting(cut(token(text)))
    print(text)
    postags = dependence_parser.pyltp_postagger(words)
    netags = dependence_parser.pyltp_ner(words, postags)
    arcs = dependence_parser.pyltp_parser(words, postags)
    roles = dependence_parser.pyltp_role_parsring(words, postags, arcs)

    if roles:

        for role in roles:
            if words[role.index] in say_word.keys():
                if say_word[words[role.index]] > 5:
                    # print (words[role.index], "".join(
                    # ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))

                    people = []
                    verb = []
                    contents = []
                    for role in roles:
                        if words[role.index] in say_word.keys():
                            if say_word[words[role.index]] > 5:
                                parsing_argument = []
                                for arg in role.arguments:
                                    parsing_argument.append(arg.name)
                                if 'A0' in parsing_argument and 'A1' in parsing_argument:
                                    verb.append(words[role.index])
                                    for arg in role.arguments:
                                        if arg.name == 'A0':
                                            people.append(
                                                ''.join(words[i] for i in range(arg.range.start, arg.range.end + 1)))
                                        if arg.name == 'A1':
                                            contents.append(
                                                ''.join(words[i] for i in range(arg.range.start, arg.range.end + 1)))
    else:
        return None, None
    table = []

    for i in range(len(people)):
        table.append([people[i], verb[i], contents[i]])
    wc = WordCloud(background_color='white', font_path='D://senior/aiCourse/dataSource/SourceHanSerifSC-Regular.otf')
    wc.generate_from_text(' '.join(words))
    wc.to_file(ROOT_DIR+'/app/static/images/comment_wc.png')
    return table,ROOT_DIR+'/app/static/images/comment_wc.png'

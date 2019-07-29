import os
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from pyltp import Parser
from pyltp import SementicRoleLabeller

LTP_DATA_DIR = 'D://senior/aiCourse/dataSource/ltp_data_v3.4.0/'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
srl_model_path = os.path.join(LTP_DATA_DIR, 'pisrl_win.model')


def pyltp_cutting(sentence):
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    result = segmentor.segment(sentence)  # 分词
    #print ('\t'.join(words))
    segmentor.release()  # 释放模型
    return result

def pyltp_postagger(words):
    postagger = Postagger() # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    result = postagger.postag(words)  # 词性标注

   # print ('\t'.join(postags))
    postagger.release()  # 释放模型
    return result
def pyltp_ner(words,postags):
    recognizer = NamedEntityRecognizer() # 初始化实例
    recognizer.load(ner_model_path)  # 加载模型


    result = recognizer.recognize(words, postags)  # 命名实体识别

    #print ('\t'.join(netags))
    recognizer.release()  # 释放模型
    return result

def pyltp_parser(words,postags):
    parser = Parser() # 初始化实例
    parser.load(par_model_path)  # 加载模型
    result = parser.parse(words, postags)  # 句法分析

    #print ("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    parser.release()  # 释放模型
    return result

def pyltp_role_parsring(words, postags, arcs):
    labeller = SementicRoleLabeller() # 初始化实例
    labeller.load(srl_model_path)  # 加载模型
    result = labeller.label(words, postags, arcs)  # 语义角色标注
    #for role in roles:
    #    print (words[role.index], "".join(
    #        ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
    labeller.release()  # 释放模型
    return result
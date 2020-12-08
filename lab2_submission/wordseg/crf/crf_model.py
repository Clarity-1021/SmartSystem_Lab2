from lab2_submission.wordseg.tools.tools import *
from lab2_submission.wordseg.crf.crf import *


def predict(sentence):
    TEMPLATE_PATH = 'wordseg/crf/' + 'template_4.utf8'
    SAVE_NAME = '10_4_U'
    LAB_NUM = '1'
    EPOCH = '8'
    RIGHT_RATE = '73.273'
    B_LOAD_PATH = 'wordseg/crf/' + SAVE_NAME + '/' + LAB_NUM + '/b_trained_' + EPOCH + '_' + RIGHT_RATE + '%.py'
    Uni_t = []
    Bi_t = []
    B = load_data(B_LOAD_PATH)

    # 导入模板
    with open(TEMPLATE_PATH, encoding='utf-8') as tp_file:
        templates = tp_file.read().split('\n\n')
        uni_gram = templates[0].split()
        load_template(uni_gram, Uni_t)
        bi_gram = templates[1].split()
        load_template(bi_gram, Bi_t)

    return predict_sentence(sentence, Uni_t, B)

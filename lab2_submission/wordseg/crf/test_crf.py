from lab2_submission.wordseg.crf.crf import *
from lab2_submission.wordseg.tools.tools import *

if __name__ == '__main__':

    TRAIN_DATASET_NUM = '30'
    LAB_NUM = '1/'
    TEMPLATE_NUM = '10'
    DATA_PATH = 'dataset_' + TRAIN_DATASET_NUM + '/'
    SAVE_NAME = 'U_' + TEMPLATE_NUM + '/'
    TEST_PATH = DATA_PATH + 'test/test1.utf8'
    TEMPLATE_PATH = 'templates/template_' + TEMPLATE_NUM + '.utf8'
    EPOCH = '91'
    RIGHT_RATE = '91.032'
    B_LOAD_PATH = DATA_PATH + SAVE_NAME + LAB_NUM + 'b_trained_' + EPOCH + '_' + RIGHT_RATE + '%.py'

    Uni_t = []
    Bi_t = []
    print('load features...')
    B = load_data(B_LOAD_PATH)

    print('load template...')
    # 导入模板
    with open(TEMPLATE_PATH, encoding='utf-8') as tp_file:
        templates = tp_file.read().split('\n\n')
        uni_gram = templates[0].split()
        load_template(uni_gram, Uni_t)
        bi_gram = templates[1].split()
        load_template(bi_gram, Bi_t)

    print('load data...')
    # 导入数据
    with open(TEST_PATH, encoding='utf-8') as file:
        test_sentences = file.read().split('\n\n')

    print('start predict...')
    # test_right_rate = predict_test_set(test_sentences, Uni_t, B)
    for (index, sentence) in enumerate(test_sentences):
        test_tuple = sentence.split()
        print('[', index, ']:')
        print('句子=', test_tuple[0])
        print('实际=', test_tuple[1])
        print('预测=', predict_sentence(test_tuple[0], Uni_t, B))
    # print('test_right_rate = %.3f%%' % (test_right_rate * 100))
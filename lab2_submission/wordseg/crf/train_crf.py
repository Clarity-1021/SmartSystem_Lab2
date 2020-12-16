from lab2_submission.wordseg.tools.tools import *
from lab2_submission.wordseg.crf.crf import *

if __name__ == '__main__':

    TRAIN_DATASET_NUM = '30'
    LAB_NUM = '1/'
    TEMPLATE_NUM = '11'
    DATA_PATH = 'dataset_' + TRAIN_DATASET_NUM + '/'
    SAVE_NAME = 'U_' + TEMPLATE_NUM + '/'
    TRAIN_PATH = DATA_PATH + 'train/train.utf8'
    DEVELOP_PATH = DATA_PATH + 'develop/develop.utf8'
    TEMPLATE_PATH = 'templates/template_' + TEMPLATE_NUM + '.utf8'
    B_SAVE_PATH = DATA_PATH + SAVE_NAME + LAB_NUM + 'b_trained_'
    # B_LOAD_PATH = DATA_PATH + SAVE_NAME + '1/b_trained_53_90.461%.py'
    B_LOAD_PATH = DATA_PATH + SAVE_NAME + 'b.py'
    RT_SAVE_PATH = DATA_PATH + SAVE_NAME + LAB_NUM + 'train_right_rate.py'
    RT_D_SAVE_PATH = DATA_PATH + SAVE_NAME + LAB_NUM + 'develop_right_rate.py'
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
    with open(TRAIN_PATH, encoding='utf-8') as file:
        sentences = file.read().split('\n\n')

    with open(DEVELOP_PATH, encoding='utf-8') as file:
        develop_sentences = file.read().split('\n\n')

    print('start train...')
    max_right = 0.0
    develop_right_rates = []
    right_rates = []
    for epoch in range(1, 500):
        print('epoch[%d] start...' % (epoch))
        total_right_rate = 0.0
        for sentence in sentences:
            lines = sentence.split('\n')
            sentence_len = len(lines)
            right_count = predict_feature_gram(lines, Uni_t, B)
            # right_count = predict_feature_gram(lines, Bi_t)
            right_rate = right_count / sentence_len
            total_right_rate += right_rate
            # print('%.3f%%' % (right_rate * 100))
        total_right_rate /= len(sentences)
        right_rates.append(total_right_rate)
        develop_right_rate = predict_test_set(develop_sentences, Uni_t, B)
        develop_right_rates.append(develop_right_rate)
        if develop_right_rate > max_right:
            max_right = develop_right_rate
            save_data(B_SAVE_PATH + str(epoch) + '_%.3f%%.py' % (develop_right_rate * 100), B)
        save_data(RT_SAVE_PATH, right_rates)
        save_data(RT_D_SAVE_PATH, develop_right_rates)
        print('train_right_rate = %.3f%%' % (total_right_rate * 100))
        print('develop_right_rate = %.3f%%' % (develop_right_rate * 100))

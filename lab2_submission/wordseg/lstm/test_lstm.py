from lab2_submission.wordseg.lstm.batch.model import *
from lab2_submission.wordseg.tools.tools import *


if __name__ == '__main__':

    DATASET_NUM = '7'  # 训练集和开发集的文件夹号
    LAB_NUM = '3'  # 实验号
    MODLE_SAVE_PATH = DATASET_NUM + '/bi_lstm_crf_model_' + LAB_NUM + '.pth'
    WTI_SAVE_PATH = DATASET_NUM + '/word_to_ix_' + LAB_NUM + '.py'
    TEST_PATH = DATASET_NUM + '/train_set.py'

    print('load model...')
    model = torch.load(MODLE_SAVE_PATH)  # 加载模型
    word_to_ix = load_data(WTI_SAVE_PATH)  # 加载字典
    tag_to_ix = {"S": 0, "B": 1, "I": 2, "E": 3, START_TAG: 4, STOP_TAG: 5, PAD_TAG: 6}

    print('load test set...')
    test_data = load_data(TEST_PATH)  # 加载测试集

    print('predict...')
    # 预测
    test_right_rate = predict_right_rate(test_data, model, word_to_ix, tag_to_ix)
    print('test_right_rate=%.3f%%' % (test_right_rate * 100))






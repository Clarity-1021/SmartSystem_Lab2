from lab2_submission.wordseg.lstm.batch.model import *
from lab2_submission.wordseg.tools.tools import *


def predict(sentence):

    DATASET_NUM = '9'  # 训练集和开发集的文件夹号
    LAB_NUM = '2'  # 实验号
    MODLE_SAVE_PATH = 'wordseg/lstm/' + DATASET_NUM + '/bi_lstm_crf_model_source_' + LAB_NUM + '.pth'
    WTI_SAVE_PATH = 'wordseg/lstm/' + DATASET_NUM + '/word_to_ix_source_' + LAB_NUM + '.py'

    model = torch.load(MODLE_SAVE_PATH)  # 加载模型
    word_to_ix = load_data(WTI_SAVE_PATH)  # 加载字典
    ix_to_tag = ['S', 'B', 'I', 'E']

    precheck_sent = prepare_sequence(list(sentence), word_to_ix)
    (tensor_num, predict_tags) = model(precheck_sent)
    predict_result = [ix_to_tag[ix] for ix in predict_tags]

    return ''.join(predict_result)
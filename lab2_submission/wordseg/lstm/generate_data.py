from lab2_submission.wordseg.tools.tools import *

DATASET_NUM = '6'
SAVE_NUM = '9'
TRAIN_PATH = '../../../dataset/dataset' + DATASET_NUM + '/train.utf8'
SAVE_DATA_SET_PATH = SAVE_NUM + '/develop_set.py'
data_set = []

print('generate data...')
with open(TRAIN_PATH, encoding='utf-8') as file:
    sentences = file.read().split('\n\n')
    for sentence in sentences:
        lines = sentence.split('\n')
        words = []
        states = []
        for line in lines:
            word = line.split()[0]
            state = line.split()[1]
            words += [word]
            states += [state]
        data_set.append((words, states))
    pass


print('save data...')
# 存语料集
save_data(SAVE_DATA_SET_PATH, data_set)
print('got data.')
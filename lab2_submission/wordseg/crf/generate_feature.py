import numpy as np


# 存数据
def save_data(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as b_file:
        print(data, file=b_file)


DATASET_NUM = '5'
TRAIN_PATH = '../../../dataset/dataset' + DATASET_NUM + '/train.utf8'
TEMPLATE_PATH = '../../../dataset/dataset' + DATASET_NUM + '/template.utf8'
B_SAVE_PATH = DATASET_NUM + '/b.py'

S = ['S', 'B', 'I', 'E']  # 状态
Uni_t = []
Bi_t = []
B = []
WINDOW_SIZE = 5


def load_template(gram, template_list):
    for i in range(2, len(gram)):
        cur_tp = gram[i].split(':')[1].split('/')
        tp_list = []
        for j in range(len(cur_tp)):
            tp = cur_tp[j].split(',')
            tp_list.append((int(tp[0][3:]), int(tp[1][:-1])))
        template_list.append(tp_list)


with open(TEMPLATE_PATH, encoding='utf-8') as tp_file:
    templates = tp_file.read().split('\n\n')
    uni_gram = templates[0].split()
    load_template(uni_gram, Uni_t)
    bi_gram = templates[1].split()
    load_template(bi_gram, Bi_t)


def new_S_dic():
    result = {}
    for state in S:
        result[state] = {}
    return result


for i in range(len(Uni_t)):
    B.append(new_S_dic())


def update_cur_level_feature(dic, attribute, lines, pos_index):
    if attribute == 0:
        pos_attribute = lines[pos_index].split()[0]
    else:
        pos_attribute = lines[pos_index].split()[2]
    if pos_attribute not in dic:
        dic[pos_attribute] = {}
    pass
    return dic[pos_attribute]


def update_window_feature_Uni_gram(lines, start_index, window_size):
    for i in range(window_size):
        cur_index = start_index + i
        cur_state = lines[cur_index].split()[1]
        for template_index in range(len(Uni_t)):
            cur_level_dic = B[template_index][cur_state]
            template = Uni_t[template_index]
            for sub_tp_index in range(len(template)):
                (pos, attribute) = template[sub_tp_index]
                if pos + i >= 0 and pos + i < window_size:
                    pos_index = pos + cur_index
                    if sub_tp_index < len(template) - 1:
                        cur_level_dic = update_cur_level_feature(cur_level_dic, attribute, lines, pos_index)
                    else:
                        if attribute == 0:
                            pos_attribute = lines[pos_index].split()[0]
                        else:
                            pos_attribute = lines[pos_index].split()[2]
                        if pos_attribute not in cur_level_dic:
                            cur_level_dic[pos_attribute] = 0
                        cur_level_dic[pos_attribute] += 1
                else:
                    break
                pass
            pass
        pass


# 导入数据
with open(TRAIN_PATH, encoding='utf-8') as file:
    sentences = file.read().split('\n\n')
    for sentence in sentences:
        lines = sentence.split('\n')
        sentence_len = len(lines)
        epoch = int(sentence_len / WINDOW_SIZE)
        remain = sentence_len % WINDOW_SIZE
        for epoch_index in range(epoch):
            start_index = epoch_index * WINDOW_SIZE
            update_window_feature_Uni_gram(lines, start_index, WINDOW_SIZE)
            pass
        if remain != 0:
            start_index = epoch * WINDOW_SIZE
            update_window_feature_Uni_gram(lines, start_index, remain)
print('statistics to complete.')

save_data(B_SAVE_PATH, B)
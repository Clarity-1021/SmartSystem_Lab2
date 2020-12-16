from lab2_submission.wordseg.tools.tools import *

DATASET_NUM = '30'
# SAVE_NUM = '9'
TRAIN_PATH = 'dataset_' + DATASET_NUM + '/train/train.utf8'
TEMPLATE_NUM = '11'
TEMPLATE_PATH = 'templates/template_' + TEMPLATE_NUM + '.utf8'
SAVE_NAME = 'U_' + TEMPLATE_NUM
B_SAVE_PATH = 'dataset_' + DATASET_NUM + '/' + SAVE_NAME + '/b.py'
data_set = []


S = ['S', 'B', 'I', 'E']  # 状态z
Uni_t = []
Bi_t = []
B = []


def update_feature_gram(lines, gram):
    sent_len = len(lines)
    for cur_index in range(sent_len):
        cur_state = lines[cur_index].split()[1]
        for template_index in range(len(Uni_t)):
            cur_dic = B[template_index][cur_state]
            template = gram[template_index]
            pos_attribute = ''
            template_len = len(template)
            for sub_tp_index in range(template_len):
                (pos, attribute) = template[sub_tp_index]
                if 0 <= pos + cur_index < sentence_len:
                    pos_index = cur_index + pos
                    if attribute == 0:
                        pos_attribute += lines[pos_index].split()[0]
                    else:
                        pos_attribute += lines[pos_index].split()[1]
                else:
                    break
            if len(pos_attribute) == template_len:
                if pos_attribute in cur_dic:
                    cur_dic[pos_attribute] += 1
                else:
                    cur_dic[pos_attribute] = 1


# 导入模板
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

# 导入数据
with open(TRAIN_PATH, encoding='utf-8') as file:
    sentences = file.read().split('\n\n')
    # for sentence in sentences:
    k = 0
    for i in range(len(sentences)):
        sentence = sentences[i]
        lines = sentence.split('\n')
        sentence_len = len(lines)
        k += sentence_len
        k+=1
        update_feature_gram(lines, Uni_t)
        # update_feature_gram(lines, Bi_t)
print('statistics to complete.')

save_data(B_SAVE_PATH, B)
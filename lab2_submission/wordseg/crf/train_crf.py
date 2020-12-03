import matplotlib.pyplot as plt


# 存数据
def save_data(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as b_file:
        print(data, file=b_file)


# 读数据
def load_data(file_name):
    file = open(file_name,'rb').read()
    file = file.decode(encoding='UTF-8',errors='strict')
    return eval(file)


TRAIN_DATASET_NUM = '1'
DEVELOP_DATASET_NUM = '6'
TRAIN_NUM = '2'
TRAIN_PATH = '../../../dataset/dataset' + TRAIN_DATASET_NUM + '/train.utf8'
DEVELOP_PATH = '../../../dataset/dataset' + DEVELOP_DATASET_NUM + '/develop.utf8'
TEMPLATE_PATH = '../../../dataset/dataset' + TRAIN_DATASET_NUM + '/template.utf8'
B_SAVE_PATH = TRAIN_DATASET_NUM + '/b_trained_' + TRAIN_NUM + '.py'
B_LOAD_PATH = TRAIN_DATASET_NUM + '/b.py'
RT_SAVE_PATH = TRAIN_DATASET_NUM + '/train_right_rate_' + TRAIN_NUM + '.py'
RT_D_SAVE_PATH = TRAIN_DATASET_NUM + '/develop_right_rate_' + TRAIN_NUM + '.py'

S = ['S', 'B', 'I', 'E']  # 状态
Uni_t = []
Bi_t = []
B = load_data(B_LOAD_PATH)
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


def new_V_dic():
    result = {}
    for state in S:
        result[state] = 0
    return result


def update_cur_level_feature(dic, attribute, lines, pos_index):
    if attribute == 0:
        pos_attribute = lines[pos_index].split()[0]
    else:
        pos_attribute = lines[pos_index].split()[2]
    pass
    if pos_attribute in dic:
        return dic[pos_attribute]
    return False


def update_feature(wrong_state, i, lines, window_size):
    cur_index = start_index + i
    real_state = lines[cur_index].split()[1]
    for template_index in range(len(Uni_t)):
        cur_level_dic = B[template_index][real_state]
        template = Uni_t[template_index]
        for sub_tp_index in range(len(template)):
            (pos, attribute) = template[sub_tp_index]
            if 0 <= pos + i < window_size:
                pos_index = pos + cur_index
                if sub_tp_index < len(template) - 1:
                    cur_level_dic = update_cur_level_feature(cur_level_dic, attribute, lines, pos_index)
                    if not cur_level_dic:
                        break
                else:
                    if attribute == 0:
                        pos_attribute = lines[pos_index].split()[0]
                    else:
                        pos_attribute = lines[pos_index].split()[2]
                    if pos_attribute in cur_level_dic:
                        cur_level_dic[pos_attribute] += 1
            else:
                break
            pass
        pass
    for template_index in range(len(Uni_t)):
        cur_level_dic = B[template_index][wrong_state]
        template = Uni_t[template_index]
        for sub_tp_index in range(len(template)):
            (pos, attribute) = template[sub_tp_index]
            if 0 <= pos + i < window_size:
                pos_index = pos + cur_index
                if sub_tp_index < len(template) - 1:
                    cur_level_dic = update_cur_level_feature(cur_level_dic, attribute, lines, pos_index)
                    if not cur_level_dic:
                        break
                else:
                    if attribute == 0:
                        pos_attribute = lines[pos_index].split()[0]
                    else:
                        pos_attribute = lines[pos_index].split()[2]
                    if pos_attribute in cur_level_dic:
                        cur_level_dic[pos_attribute] = max(0, cur_level_dic[pos_attribute] - 1)
            else:
                break
            pass
        pass
    pass


def predict_window_feature_Uni_gram(lines, start_index, window_size):
    path = []
    result = 0
    for i in range(window_size):
        cur_index = start_index + i
        real_state = lines[cur_index].split()[1]
        V = new_V_dic()
        for cur_state in S:
            for template_index in range(len(Uni_t)):
                cur_level_dic = B[template_index][cur_state]
                template = Uni_t[template_index]
                for sub_tp_index in range(len(template)):
                    (pos, attribute) = template[sub_tp_index]
                    if 0 <= pos + i < window_size:
                        pos_index = pos + cur_index
                        if sub_tp_index < len(template) - 1:
                            cur_level_dic = update_cur_level_feature(cur_level_dic, attribute, lines, pos_index)
                            if not cur_level_dic:
                                break
                        else:
                            if attribute == 0:
                                pos_attribute = lines[pos_index].split()[0]
                            else:
                                pos_attribute = lines[pos_index].split()[2]
                            V[cur_state] += cur_level_dic.get(pos_attribute, 0)
                    else:
                        break
                    pass
                pass
        pass
        max_index = 0
        for S_index in range(1, len(S)):
            if V[S[S_index]] > V[S[max_index]]:
                max_index = S_index
        path.append(S[max_index])
        if real_state == S[max_index]:
            result += 1
        else:
            update_feature(S[max_index], i, lines, window_size)
    # print(path)
    return result


# 导入数据
with open(TRAIN_PATH, encoding='utf-8') as file:
    sentences = file.read().split('\n\n')


with open(DEVELOP_PATH, encoding='utf-8') as file:
    develop_sentences = file.read().split('\n\n')


def predict_window_develop(lines, start_index, window_size):
    path = []
    result = 0
    for i in range(window_size):
        cur_index = start_index + i
        real_state = lines[cur_index].split()[1]
        V = new_V_dic()
        for cur_state in S:
            for template_index in range(len(Uni_t)):
                cur_level_dic = B[template_index][cur_state]
                template = Uni_t[template_index]
                for sub_tp_index in range(len(template)):
                    (pos, attribute) = template[sub_tp_index]
                    if 0 <= pos + i < window_size:
                        pos_index = pos + cur_index
                        if sub_tp_index < len(template) - 1:
                            cur_level_dic = update_cur_level_feature(cur_level_dic, attribute, lines, pos_index)
                            if not cur_level_dic:
                                break
                        else:
                            if attribute == 0:
                                pos_attribute = lines[pos_index].split()[0]
                            else:
                                pos_attribute = lines[pos_index].split()[2]
                            V[cur_state] += cur_level_dic.get(pos_attribute, 0)
                    else:
                        break
                    pass
                pass
        pass
        max_index = 0
        for S_index in range(1, len(S)):
            if V[S[S_index]] > V[S[max_index]]:
                max_index = S_index
        path.append(S[max_index])
        if real_state == S[max_index]:
            result += 1
    # print(path)
    return result


def predict():
    total_right_rate = 0.0
    for sentence in develop_sentences:
        right_count = 0
        lines = sentence.split('\n')
        sentence_len = len(lines)
        epoch = int(sentence_len / WINDOW_SIZE)
        remain = sentence_len % WINDOW_SIZE
        for epoch_index in range(epoch):
            start_index = epoch_index * WINDOW_SIZE
            right_count += predict_window_develop(lines, start_index, WINDOW_SIZE)
            pass
        if remain != 0:
            start_index = epoch * WINDOW_SIZE
            right_count += predict_window_develop(lines, start_index, remain)
        right_rate = right_count / sentence_len
        total_right_rate += right_rate
        # print('%.3f%%' % (right_rate * 100))
    total_right_rate /= len(develop_sentences)
    return total_right_rate
    # pass

develop_right_rates = []
right_rates = []
for kk in range(1, 300):
    print('epoch[%2d]' % (kk))
    total_right_rate = 0.0
    for sentence in sentences:
        right_count = 0
        lines = sentence.split('\n')
        sentence_len = len(lines)
        epoch = int(sentence_len / WINDOW_SIZE)
        remain = sentence_len % WINDOW_SIZE
        for epoch_index in range(epoch):
            start_index = epoch_index * WINDOW_SIZE
            right_count += predict_window_feature_Uni_gram(lines, start_index, WINDOW_SIZE)
            pass
        if remain != 0:
            start_index = epoch * WINDOW_SIZE
            right_count += predict_window_feature_Uni_gram(lines, start_index, remain)
        right_rate = right_count / sentence_len
        total_right_rate += right_rate
        # print('%.3f%%' % (right_rate * 100))
    total_right_rate /= len(sentences)
    right_rates.append(total_right_rate)
    develop_right_rate = predict()
    develop_right_rates.append(develop_right_rate)
    save_data(B_SAVE_PATH, B)
    save_data(RT_SAVE_PATH, right_rates)
    save_data(RT_D_SAVE_PATH, develop_right_rates)
    print('train_right_rate = %.3f%%' % (total_right_rate * 100))
    print('develop_right_rate = %.3f%%' % (develop_right_rate * 100))
    print('statistics to complete.')


plt.figure()
plt.plot(right_rates)
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('right_rate')
plt.show()

save_data(B_SAVE_PATH, B)
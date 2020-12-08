S = ['B', 'I', 'E', 'S']  # 状态
# S = ['S', 'B', 'I', 'E']  # 状态

def new_V_dic():
    result = {}
    for state in S:
        result[state] = 0
    return result


def new_path_dic(first_winner):
    result = {}
    for state in S:
        result[state] = [first_winner]
    return result


def update_trans_feature(real_state, wrong_state, cur_state, B):
    cur_dic = B[-1][cur_state]
    if real_state in cur_dic:
        cur_dic[real_state] += 1
    else:
        print('np-1.')
    if wrong_state in cur_dic:
        cur_dic[wrong_state] -= 1
        # cur_dic[wrong_state] = max(0, cur_dic[wrong_state] - 1)
    else:
        cur_dic[wrong_state] = -1


def update_feature(real_state, wrong_state, lines, cur_index, gram, B):
    sent_len = len(lines)
    for template_index in range(len(gram)):
        cur_dic = B[template_index][real_state]
        template = gram[template_index]
        pos_attribute = ''
        template_len = len(template)
        for sub_tp_index in range(template_len):
            (pos, attribute) = template[sub_tp_index]
            if 0 <= pos + cur_index < sent_len:
                pos_index = cur_index + pos
                if attribute == 0:
                    pos_attribute += lines[pos_index].split()[0]
                else:
                    pos_attribute = lines[pos_index].split()[1]
            else:
                break
        if len(pos_attribute) == template_len:
            if pos_attribute in cur_dic:
                cur_dic[pos_attribute] += 1
                # cur_dic[pos_attribute] = max(0, cur_level_dic[pos_attribute] - 1)
            else:
                cur_dic[pos_attribute] = 1
    for template_index in range(len(gram)):
        cur_dic = B[template_index][wrong_state]
        template = gram[template_index]
        pos_attribute = ''
        template_len = len(template)
        for sub_tp_index in range(template_len):
            (pos, attribute) = template[sub_tp_index]
            if 0 <= pos + cur_index < sent_len:
                pos_index = cur_index + pos
                if attribute == 0:
                    pos_attribute += lines[pos_index].split()[0]
                else:
                    pos_attribute = lines[pos_index].split()[1]
            else:
                break
        if len(pos_attribute) == template_len:
            if pos_attribute in cur_dic:
                cur_dic[pos_attribute] -= 1
                # cur_dic[pos_attribute] = max(0, cur_level_dic[pos_attribute] - 1)
            else:
                cur_dic[pos_attribute] = -1


def predict(lines, gram, B):
    path = {}
    V_last = new_V_dic()
    V = {}
    O = []
    sent_len = len(lines)
    for cur_index in range(sent_len):
        real_state = lines[cur_index].split()[1]
        O.append(real_state)
        V = {}
        new_path = {}
        for cur_state in S:
            V_list = []
            for last_state in S:
                new_count = V_last.get(last_state, 0)
                for template_index in range(len(gram)):
                    cur_dic = B[template_index][cur_state]
                    template = gram[template_index]
                    pos_attribute = ''
                    template_len = len(template)
                    for sub_tp_index in range(template_len):
                        (pos, attribute) = template[sub_tp_index]
                        if 0 <= pos + cur_index < sent_len:
                            pos_index = cur_index + pos
                            if attribute == 0:
                                pos_attribute += lines[pos_index].split()[0]
                            else:
                                pos_attribute += path[last_state][pos]
                        else:
                            break
                    if len(pos_attribute) == template_len:
                        new_count += cur_dic.get(pos_attribute, 0)
                V_list.append((new_count, last_state))
            # print(V_list)
            (weight_count, max_last_state) = max(V_list)
            V[cur_state] = weight_count
            if cur_index > 0:
                new_path[cur_state] = path[max_last_state] + [cur_state]
        pass
        V_last = V
        if cur_index == 0:
            path = new_path_dic(get_winner(V_last))
        else:
            path = new_path
    winner = get_winner(V)
    predict_order = path[winner]
    return O, predict_order


def predict_sentence(sentence, gram, B):
    path = {}
    V_last = new_V_dic()
    V = {}
    sent_len = len(sentence)
    for cur_index in range(sent_len):
        V = {}
        new_path = {}
        for cur_state in S:
            V_list = []
            for last_state in S:
                new_count = V_last.get(last_state, 0)
                for template_index in range(len(gram)):
                    cur_dic = B[template_index][cur_state]
                    template = gram[template_index]
                    pos_attribute = ''
                    template_len = len(template)
                    for sub_tp_index in range(template_len):
                        (pos, attribute) = template[sub_tp_index]
                        if 0 <= pos + cur_index < sent_len:
                            pos_index = cur_index + pos
                            if attribute == 0:
                                pos_attribute += sentence[pos_index]
                            else:
                                pos_attribute += path[last_state][pos]
                        else:
                            break
                    if len(pos_attribute) == template_len:
                        new_count += cur_dic.get(pos_attribute, 0)
                V_list.append((new_count, last_state))
            # print(V_list)
            (weight_count, max_last_state) = max(V_list)
            V[cur_state] = weight_count
            if cur_index > 0:
                new_path[cur_state] = path[max_last_state] + [cur_state]
        pass
        V_last = V
        if cur_index == 0:
            path = new_path_dic(get_winner(V_last))
        else:
            path = new_path
    winner = get_winner(V)
    predict_order = path[winner]
    return ''.join(predict_order)


def get_winner(dic):
    (prob, state) = max((dic[s], s) for s in S)
    return state


def predict_feature_gram(lines, gram, B):
    result = 0
    O, P = predict(lines, gram, B)
    sent_len = len(lines)
    # print(''.join(predict_order))
    # print(''.join(O))
    for i in range(sent_len):
        if P[i] == O[i]:
            result += 1
        else:
            update_feature(O[i], P[i], lines, i, gram, B)
            if i + 1 < sent_len:
                update_trans_feature(O[i], P[i], O[i + 1], B)
    return result


def predict_test(lines, gram, B):
    result = 0
    O, P = predict(lines, gram, B)
    sent_len = len(lines)
    # print(''.join(predict_order))
    # print(''.join(O))
    for i in range(sent_len):
        if P[i] == O[i]:
            result += 1
    return result


def predict_test_set(develop_sentences, gram, B):
    total_right_rate = 0.0
    for sentence in develop_sentences:
        lines = sentence.split('\n')
        sentence_len = len(lines)
        right_count = predict_test(lines, gram, B)
        # right_count = predict_develop(lines, Bi_t)
        right_rate = right_count / sentence_len
        total_right_rate += right_rate
        # print('%.3f%%' % (right_rate * 100))
    total_right_rate /= len(develop_sentences)
    return total_right_rate
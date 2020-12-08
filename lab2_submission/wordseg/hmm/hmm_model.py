DATASET_NUM = '3'
# PI_SAVE_PATH = DATASET_NUM + '/pi.py'
# A_SAVE_PATH = DATASET_NUM + '/a.py'
# B_SAVE_PATH = DATASET_NUM + '/b.py'
# K_SAVE_PATH = DATASET_NUM + '/k.py'

PI_SAVE_PATH = 'wordseg/hmm/' + DATASET_NUM + '/pi.py'
A_SAVE_PATH = 'wordseg/hmm/' + DATASET_NUM + '/a.py'
B_SAVE_PATH = 'wordseg/hmm/' + DATASET_NUM + '/b.py'
K_SAVE_PATH = 'wordseg/hmm/' + DATASET_NUM + '/k.py'

S = ['S', 'B', 'I', 'E']  # Set of states


def load_data(file_name):
    file = open(file_name,'rb').read()
    file = file.decode(encoding='UTF-8',errors='strict')
    return eval(file)


PI = load_data(PI_SAVE_PATH)
A = load_data(A_SAVE_PATH)
B = load_data(B_SAVE_PATH)
K = load_data(K_SAVE_PATH)

# sentence = '今天天气怎么样'
# sentence = '《东方书林之旅》告诉你什么叫出版策划——书海弄潮儿'
# sentence = '构成了近年来中国图书市场最抢眼的一道风景线。'
# sentence = '从１９９２年人民出版社成立策划室并由方鸣任主任开始，作为出版策划的倡言者，方鸣搞了一系列的出版策划行动，其中首推最近几年陆续出版的令出版界和读书界为之振奋的《东方书林之旅》。这套由６个书系、２４种学术文化精品图书组成的系列丛书，构成了近年来中国图书市场最抢眼的一道风景线。'
# sentence = '方鸣认为，策划是对图书进行整体包装，不单单是指书稿内容和装帧设计，还包括市场等。他认为，图书也像人穿衣一样，不仅料子要好，还要做工精致。只有追求精致、精道、精当、精美，才能反映出书的内在价值。譬如《东方书林之旅》，首次采用黄色胶板纸作为内瓤，柔和的米黄色给人一种读书人的儒雅感觉，油然生出一种淡淡的书香气；外封以高级乌光铜板纸精制而成，追求高品质；以优良的牛皮纸制作内封，通过“牛皮纸情结”，使那些从小就习惯用牛皮纸包书皮的学子们产生怀旧感和亲近感。'
# sentence = '作为出版家的方鸣有着现代商人的头脑，他说他绝不单纯为赚钱而出书，然而他出的书又常常赚钱。在图书市场竞争激烈并已形成买方市场的今天，别人对学术图书避之犹恐不及，他却不出热门专出冷门，在学术界大肆招兵买马，开风气之先。方鸣说，潮流是人为创造的，人要领着命运走，不能被命运拖着走，不能随波逐流。当今中国，有一个独特的读者群，他们追求的是诗意的、有性灵的东西，在当代都市文化的喧嚣中保有非常独特的精神品位、价值取向和审美趣味。虽然这些人分散来看不多，然而聚合起来也就是一大批。他们是这些纯学术、纯文化、高规格、高品位图书的知音。方鸣的搭档、《东方书林之旅》主持刘丽华说，他们出的书常常不是一下子就赚很多钱，而是慢慢让人喜爱，慢慢走红和畅销。（崔伟）'
# sentence = '农民与大师'
# sentence = '丰子恺当年送他三句话：“多读书，广结交，少说话”——－（附图片１张）'
# sentence = '胡世庆出生在浙江省绍兴县湖塘镇西跨湖村，因家贫，初中未毕业就辍学务农。他省吃俭用去买了许多他喜爱的文史方面的书，还大量摘抄他认为有用的材料。不知道是什么书上的什么话激发起了他的一阵冲动，使他不顾一切拿起笔，给当时在上海的文学家丰子恺写了一封信，诉说了自己的心愿。不久，他竟收到了丰子恺先生的回信。即使在丰子恺被揪斗、被下放的时候，两人仍保持着联系，书信往还持续了１４年。丰子恺在１９７４年专门为他作了一幅斗方毛笔画：“种瓜得瓜”，并题词：“世庆贤台雅赏”（见图）。如今这幅珍贵的墨存还挂在胡世庆一家三口１０平方米左右的斗室中。丰子恺把自己几十年人生阅历浓缩成三句话送给他———“多读书，广结交，少说话”，把他引为忘年知己。'
# sentence = '胡世庆还专门去拜访一些当时被批判、受迫害的“反动”学术权威，向他们求教在读书中产生的种种疑问。他对这些学养渊厚博雅、著作等身的文史界前辈，执异常恭敬的弟子之礼。这对当时备受人格侮辱的文史界专家来说，内心的激动与热流回旋无法表达，因而对这个朴实无华的农民格外青睐。'
# sentence = '与大师们的交往请教，拓展了胡世庆的眼界。作为一个农民，他没有工薪收入，没有学历，没有职称，没有单位编制。然而，他坚持文史研究达４０年，发表学术论文２０多篇，并且已出版４６万字的《中国文化史》和８５万字的《中国文化通史》专著。如今五十开外的胡世庆已成为一个地地道道的学者。'


# 归一化
def normalization(V):
    total = 0.0
    for state in S:
        total += V[state]
    V_norm = {}
    for state in S:
        V_norm[state] = V[state] / total
    return V_norm


def get_winner(dic):
    (prob, state) = max((dic[s], s) for s in S)
    return state


# def predict():
def predict(sentence):
    sentence.encode(encoding='utf-8',errors='ignore')
    sentence_len = len(sentence)
    if sentence_len == 1:
        print('S')
        return 'S'
    V = {}
    path = {}
    # print(sentence)
    for state in S:
        V[state] = PI[state] * B[state].get(sentence[0], 0)
        path[state] = [state]
    V_norm = normalization(V)
    for t in range(1, sentence_len):
        if sentence[t] not in K:
            print('miss:' + sentence[t])
        V = {}
        new_path = {}
        for cur_state in S:
            V_list = []
            for last_state in S:
                if V_norm[last_state] > 0:
                    if sentence[t] not in K:
                        if last_state == 'S' or last_state == 'E':
                            if cur_state == 'B' or cur_state == 'S':
                                emit_prop = 0.00000001
                            else:
                                emit_prop = 0.0
                        elif last_state == 'I':
                            if cur_state == 'I' or cur_state == 'S':
                                emit_prop = 0.00000001
                            else:
                                emit_prop = 0.0
                        else:
                            if cur_state == 'I' or cur_state == 'E':
                                emit_prop = 0.00000001
                            else:
                                emit_prop = 0.0
                    else:
                        emit_prop = B[cur_state].get(sentence[t], 0)
                    cur_prop = V_norm[last_state] * A[last_state][cur_state] * emit_prop
                    V_list.append((cur_prop, last_state))
            (prob, state) = max(V_list)
            V[cur_state] = prob
            new_path[cur_state] = path[state] + [cur_state]
        V_norm = normalization(V)
        path = new_path
    winner = get_winner(V_norm)
    # print(''.join(path[winner]))
    # print('SBEBESSSBESBESBEBEBEBEBEB')
    return ''.join(path[winner])


# predict()

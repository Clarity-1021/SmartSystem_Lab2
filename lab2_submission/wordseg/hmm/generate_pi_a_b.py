DATASET_NUM = '1'
TRAIN_PATH = '../../../dataset/dataset' + DATASET_NUM + '/train.utf8'
PI_SAVE_PATH = DATASET_NUM + '/pi.py'
A_SAVE_PATH = DATASET_NUM + '/a.py'
B_SAVE_PATH = DATASET_NUM + '/b.py'
K_SAVE_PATH = DATASET_NUM + '/k.py'

S = ['S', 'B', 'I', 'E']  # 状态
A_count = {}
B_count = {}
K = set()  # 字典
PI = {}  # 初始处于该状态的概率
A = {}  # 转化概率
B = {}  # 转化发射概率
NEG_INF = -3.14e+100  # 负无限

# 初始化PI，A，B
for state_i in S:
    PI[state_i] = 0
    A_count[state_i] = 0
    B_count[state_i] = 0
    A[state_i] = {}
    B[state_i] = {}
    for state_j in S:
        A[state_i][state_j] = 0.0
        # B[state_i][state_j] = {}
        # B_count[state_i][state_j] = 0

# 导入数据
with open(TRAIN_PATH, encoding='utf-8') as file:
    sentences = file.read().split('\n\n')
    for sentence in sentences:
        line = sentence.split()
        sentence_len = int(len(line) / 2)
        X = []
        O = []
        for i in range(sentence_len):
            word = line[2*i]
            tag = line[2*i+1]
            # K.add(word)
            # word_set.add(args[0])
            X.append(tag)
            O.append(word)
        K = K | set(O)
        for i in range(len(O)):
            B_count[X[i]] += 1
            if O[i] not in B[X[i]]:
                # B[X[i - 1]][X[i]][O[i - 1]] = 0.0
                B[X[i]][O[i]] = 0.0
            # B[X[i - 1]][X[i]][O[i - 1]] += 1
            B[X[i]][O[i]] += 1
            if i == 0:
                PI[X[i]] += 1
            else:
                A_count[X[i - 1]] += 1
                # B_count[X[i - 1]][X[i]] += 1
                A[X[i - 1]][X[i]] += 1
    for state in S:
        PI[state] = PI[state] / len(sentences)
        # if PI[state] == 0:
        #     PI[state] = NEG_INF
        # else:
        #     PI[state] = PI[state] / len(sentences)
        # for word in K:
        #     if word not in B[state]:
        #         B[state][word] = 0.0


# 计算概率
# train_word_count = len(O)
# for state in S:
#     for word in K:
#         B[state][word] = 0.0

# for i in range(train_word_count - 1):
#     A[X[i]][X[i + 1]] += 1
#     B[X[i]][O[i]] += 1
# B[X[train_word_count - 1]][O[train_word_count - 1]] += 1

for state_i in S:
    for state_j in S:
        A[state_i][state_j] = A[state_i][state_j] / A_count[state_i]
    for word in K:
        if word in B[state_i]:
            B[state_i][word] = B[state_i][word] / B_count[state_i]
        #     B[state_i][word] = 0.0
        # else:

        # if A[state_i][state_j] == 0.0:
        #     A[state_i][state_j] = NEG_INF
        # else:
        #     A[state_i][state_j] = A[state_i][state_j] / S_count[state_i]

        # if B[state_i][word] == 0.0:
        #     B[state_i][word] = NEG_INF
        # else:
        #     B[state_i][word] = B[state_i][word] / S_count[state_i]



# 存PI，A，B
with open(PI_SAVE_PATH, 'w', encoding='utf-8') as pi_file:
    print(PI, file=pi_file)

with open(A_SAVE_PATH, 'w', encoding='utf-8') as a_file:
    print(A, file=a_file)

with open(B_SAVE_PATH, 'w', encoding='utf-8') as b_file:
    print(B, file=b_file)

with open(K_SAVE_PATH, 'w', encoding='utf-8') as b_file:
    print(K, file=b_file)
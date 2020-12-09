import torch.optim as optim
from lab2_submission.wordseg.lstm.batch.model import *
from lab2_submission.wordseg.tools.tools import *


if __name__ == '__main__':

    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256

    DATASET_NUM = '8'  # 训练集和开发集的文件夹号
    TRAIN_PATH = DATASET_NUM + '/train_set.py'
    DEVELOP_PATH = DATASET_NUM + '/develop_set.py'
    LAB_NUM = '2'  # 实验号
    TRAIN_RIGHT_RATES_PATH = DATASET_NUM + '/train_right_rate_' + LAB_NUM + '.py'
    DEVELOP_RIGHT_RATES_PATH = DATASET_NUM + '/develop_right_rate_' + LAB_NUM + '.py'
    MODLE_SAVE_PATH = DATASET_NUM + '/bi_lstm_crf_model_' + LAB_NUM + '.pth'
    WTI_SAVE_PATH = DATASET_NUM + '/word_to_ix_' + LAB_NUM + '.py'

    print('load train set...')
    training_data = load_data(TRAIN_PATH)  # 加载训练集

    print('load develop set...')
    develop_data = load_data(DEVELOP_PATH)  # 加载开发集

    # 字转下标
    word_to_ix = {}
    word_to_ix[PAD_TAG] = 0
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    # 标签转下标
    tag_to_ix = {"S": 0, "B": 1, "I": 2, "E": 3, START_TAG: 4, STOP_TAG: 5, PAD_TAG: 6}
    save_data(WTI_SAVE_PATH, word_to_ix)  # 存字典

    print('initialize model...')
    # 初始化模型
    model = BiLSTM_CRF_MODIFY_PARALLEL(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)  # 设置优化器

    train_right_rates = []  # 训练集的正确率
    develop_right_rates = []  # 开发集的正确率
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(1,
            300):  # again, normally you would NOT do 300 epochs, it is toy data
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        print('epoch[', epoch,'] start...')
        model.zero_grad()
        # Step 2. Get our batch inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        # If training_data can't be included in one batch, you need to sample them to build a batch
        sentence_in_pad, targets_pad = prepare_sequence_batch(training_data, word_to_ix, tag_to_ix)
        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad)
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
        print('optimizer end.')

        train_right_rate = predict_right_rate(training_data, model, word_to_ix, tag_to_ix)
        train_right_rates.append(train_right_rate)
        develop_right_rate = predict_right_rate(develop_data, model, word_to_ix, tag_to_ix)
        develop_right_rates.append(develop_right_rate)
        save_data(TRAIN_RIGHT_RATES_PATH, train_right_rates)  # 存训练集的正确率
        save_data(DEVELOP_RIGHT_RATES_PATH, develop_right_rates)  # 存开发集的正确率
        torch.save(model, MODLE_SAVE_PATH)  # 存网络
        print('train_right_rate=%.3f%%' % (train_right_rate * 100))
        print('develop_right_rate=%.3f%%' % (develop_right_rate * 100))
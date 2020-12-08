import torch.optim as optim
from lab2_submission.wordseg.tools.tools import *
from lab2_submission.wordseg.lstm.pytorch_scourse import *

if __name__ == '__main__':

    EMBEDDING_DIM = 16
    HIDDEN_DIM = 4

    DATASET_NUM = '8'
    TRAIN_PATH = DATASET_NUM + '/train_set.py'
    DEVELOP_PATH = DATASET_NUM + '/develop_set.py'
    LAB_NUM = '1'
    TRAIN_RIGHT_RATES_PATH = DATASET_NUM + '/train_right_rate_source_' + LAB_NUM + '.py'
    DEVELOP_RIGHT_RATES_PATH = DATASET_NUM + '/develop_right_rate_source_' + LAB_NUM + '.py'
    MODLE_SAVE_PATH = DATASET_NUM + '/bi_lstm_crf_model_source_' + LAB_NUM + '.pth'
    WTI_SAVE_PATH = DATASET_NUM + '/word_to_ix_source_' + LAB_NUM + '.py'

    print('load train set...')
    training_data = load_data(TRAIN_PATH)  # 加载训练集

    print('load develop set...')
    develop_data = load_data(DEVELOP_PATH)  # 加载开发集

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"S": 0, "B": 1, "I": 2, "E":3, START_TAG: 4, STOP_TAG: 5}
    save_data(WTI_SAVE_PATH, word_to_ix)  # 存字典

    print('initialize model...')
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)  # 设置优化器

    train_right_rates = []
    develop_right_rates = []
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(
            300):  # again, normally you would NOT do 300 epochs, it is toy data
        print('epoch[', epoch, '] start...')
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

        print('optimizer end.')
        train_right_rate = predict_right_rate(training_data, model, word_to_ix, tag_to_ix)
        train_right_rates.append(train_right_rate)
        develop_right_rate = predict_right_rate(develop_data, model, word_to_ix, tag_to_ix)
        develop_right_rates.append(develop_right_rate)
        save_data(TRAIN_RIGHT_RATES_PATH, train_right_rates)
        save_data(DEVELOP_RIGHT_RATES_PATH, develop_right_rates)
        torch.save(model, MODLE_SAVE_PATH)  # 存网络
        print('train_right_rate=%.3f%%' % (train_right_rate * 100))
        print('develop_right_rate=%.3f%%' % (develop_right_rate * 100))

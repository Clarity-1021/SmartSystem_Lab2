import torch.optim as optim
from lab2_submission.wordseg.tools.tools import *
from lab2_submission.wordseg.lstm.model import *

if __name__ == '__main__':

    EMBEDDING_DIM = 16
    HIDDEN_DIM = 4

    DATASET_NUM = '30'
    ROOT_PATH = 'dataset_' + DATASET_NUM + '/'
    TRAIN_PATH = ROOT_PATH + 'train_set.py'
    DEVELOP_PATH = ROOT_PATH + 'develop_set.py'
    LAB_NUM = '1_s/'
    TRAIN_RIGHT_RATES_PATH = ROOT_PATH + LAB_NUM + '/train_right_rate.py'
    TRAIN_LOSSES_PATH = ROOT_PATH + LAB_NUM + '/train_loss.py'
    DEVELOP_RIGHT_RATES_PATH = ROOT_PATH + LAB_NUM + '/develop_right_rate.py'
    DEVELOP_LOSSES_PATH = ROOT_PATH + LAB_NUM + '/develop_loss.py'
    MODEL_SAVE_PATH = ROOT_PATH + LAB_NUM + 'model_'
    WTI_SAVE_PATH = ROOT_PATH + LAB_NUM + 'word_to_ix.py'

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
    train_losses = []
    develop_right_rates = []
    develop_losses = []
    max_right_rate = 0
    train_sent_len = len(training_data)
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(1, 500):  # again, normally you would NOT do 300 epochs, it is toy data
        print('epoch[', epoch, '] start...')
        train_right_count = 0
        train_word_size = 0
        train_loss = 0
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            (tensor_num, pridict_tags) = model(sentence_in)
            train_right_count += targets.eq(torch.Tensor(pridict_tags)).numpy().sum()
            train_word_size += len(tags)
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)
            train_loss += loss
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
        print('optimize end.')
        train_right_rate = train_right_count / train_word_size
        train_loss /= train_sent_len
        # train_right_rate = predict_right_rate(training_data, model, word_to_ix, tag_to_ix)
        train_right_rates.append(train_right_rate)
        train_losses.append(train_loss)
        develop_right_rate, develop_loss = predict_right_rate(develop_data, model, word_to_ix, tag_to_ix)
        develop_right_rates.append(develop_right_rate)
        develop_losses.append(develop_loss)
        save_data(TRAIN_RIGHT_RATES_PATH, train_right_rates)
        save_data(TRAIN_LOSSES_PATH, train_losses)
        save_data(DEVELOP_RIGHT_RATES_PATH, develop_right_rates)
        save_data(DEVELOP_LOSSES_PATH, develop_losses)
        if develop_right_rate > max_right_rate:
            max_right_rate = develop_right_rate
            torch.save(model, MODEL_SAVE_PATH + '%d_%.3f%%_%.6f.pth' % (epoch, develop_right_rate * 100, develop_loss))  # 存网络
        print('train_right_rate=%.3f%% \t train_loss=%.6f' % (train_right_rate * 100, train_loss))
        print('develop_right_rate=%.3f%% \t develop_loss=%.6f' % (develop_right_rate * 100, develop_loss))

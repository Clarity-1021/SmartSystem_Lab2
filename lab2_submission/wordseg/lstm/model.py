# Author: Robert Guthrie

import torch
import torch.nn as nn

torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq]
#     return torch.tensor(idxs, dtype=torch.long)

def get_idxs(seq, to_ix):
    idxs = []
    for w in seq:
        if w not in to_ix:
            to_ix[w] = len(to_ix)
        idxs.append(to_ix[w])
    return idxs


def prepare_sequence(seq, to_ix):
    # idxs = [to_ix[w] for w in seq]
    return torch.tensor(get_idxs(seq, to_ix), dtype=torch.long)


# compute log sum exp in numerically stable way for the forward algorithm
def log_sum_exp(vec):   #vec是1*5, type是Variable
    max_score = vec[0, argmax(vec)]
    # max_score维度是１，　max_score.view(1,-1)维度是１＊１，
    # max_score.view(1, -1).expand(1, vec.size()[1])的维度是１＊５
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])

    # 里面先做减法，减去最大值可以避免e的指数次，计算机上溢
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size + 123, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

        # 预测序列的得分
        # 只是根据随机的transitions，前向传播算出的一个score
        # 用到了动态规划的思想，但因为用的是随机的转移矩阵，算出的值很大score>20

    def _forward_alg(self, feats):
        # do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)  # 1*5 而且全是-10000

        # START_TAG has all of the score
        # 因为start tag是4，所以tensor([[-10000., -10000., -10000., 0., -10000.]])，
        # 将start的值为零，表示开始进行网络的传播，
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0

        # warp in a variable so that we will get automatic backprop
        forward_var = init_alphas  # 初始状态的forward_var，随着step t变化

        # iterate through the sentence
        # 会迭代feats的行数次
        for feat in feats:  # feat的维度是５ 依次把每一行取出来~
            alphas_t = []  # the forward tensors at this timestep
            for next_tag in range(self.tagset_size):  # next tag 就是简单 i，从0到len
                # broadcast the emission(发射) score:
                # it is the same regardless of the previous tag
                # 维度是1*5 LSTM生成的矩阵是emit score
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the i_th entry of trans_score is the score of transitioning
                # to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)  # 维度是1*5
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                # 第一次迭代时理解：
                # trans_score所有其他标签到Ｂ标签的概率
                # 由lstm运行进入隐层再到输出层得到标签Ｂ的概率，emit_score维度是１＊５，5个值是相同的
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
                # 此时的alphas t 是一个长度为5，例如<class 'list'>:
                # [tensor(0.8259), tensor(2.1739), tensor(1.3526), tensor(-9999.7168), tensor(-0.7102)]
            forward_var = torch.cat(alphas_t).view(1, -1)  # 到第（t-1）step时５个标签的各自分数
        # 最后只将最后一个单词的forward var与转移 stop tag的概率相加
        # tensor([[   21.1036,    18.8673,    20.7906, -9982.2734, -9980.3135]])
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)  # alpha是一个0维的tensor

        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # 根据真实的标签算出的一个score，
    # 这与上面的def _forward_alg(self, feats)共同之处在于：
    # 两者都是用的随机转移矩阵算的score
    # 不同地方在于，上面那个函数算了一个最大可能路径，但实际上可能不是真实的各个标签转移的值
    # 例如：真实标签是N V V,但是因为transitions是随机的，所以上面的函数得到其实是N N N这样，
    # 两者之间的score就有了差距。而后来的反向传播，就能够更新transitions，使得转移矩阵逼近
    # 真实的“转移矩阵”

    # 得到gold_seq tag的score 即根据真实的label 来计算一个score，
    # 但是因为转移矩阵是随机生成的，故算出来的score不是最理想的值
    def _score_sentence(self, feats, tags):  # feats 11*5  tag 11 维
        # gives the score of a provied tag sequence
        score = torch.zeros(1)

        # 将START_TAG的标签３拼接到tag序列最前面，这样tag就是12个了
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            # feat[tags[i+1]], feat是step i 的输出结果，有５个值，
            # 对应B, I, E, START_TAG, END_TAG, 取对应标签的值
            # transition【j,i】 就是从i ->j 的转移概率值
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    # 维特比解码， 实际上就是在预测的时候使用了， 输出得分与路径值
    # 预测序列的得分
    def _viterbi_decode(self, feats):
        backpointers = []

        # initialize the viterbi variables in long space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next-tag_var[i] holds the viterbi variable for tag i
                # at the previous step, plus the score of transitioning
                # from tag i to next_tag.
                # we don't include the emission scores here because the max
                # does not depend on them(we add them in below)
                # 其他标签（B,I,E,Start,End）到标签next_tag的概率
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # now add in the emssion scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # 从step0到step(i-1)时5个序列中每个序列的最大score
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)  # bptrs_t有５个元素

        # transition to STOP_TAG
        # 其他标签到STOP_TAG的转移概率
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # follow the back pointers to decode the best path
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # pop off the start tag
        # we don't want to return that ti the caller
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()  # 把从后向前的路径正过来
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        # feats: 11*5 经过了LSTM+Linear矩阵后的输出，之后作为CRF的输入。
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


def predict_right_rate(datas, model, word_to_ix, tag_to_ix):
    right_count = 0
    word_size = 0
    average_loss = 0
    sent_len = len(datas)
    with torch.no_grad():
        for sentence, tags in datas:
            # gold_tags = data_t[1]
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.Tensor([tag_to_ix[t] for t in tags])
            (tensor_num, pridict_tags) = model(sentence_in)
            average_loss += model.neg_log_likelihood(sentence_in, targets)
            # b = torch.Tensor(pridict_tags)
            # a = precheck_tags.eq(torch.Tensor(pridict_tags)).numpy().sum()
            right_count += targets.eq(torch.Tensor(pridict_tags)).numpy().sum()
            word_size += len(tags)
    return right_count / word_size, average_loss / sent_len

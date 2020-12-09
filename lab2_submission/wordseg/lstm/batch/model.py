import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def get_idxs(seq, to_ix):
    idxs = []
    for w in seq:
        if w not in to_ix:
            to_ix[w] = len(to_ix)
        idxs.append(to_ix[w])
    return idxs

def get_idxs_pad(seqs_pad, to_ix):
    idxs_pad = []
    for seq in seqs_pad:
        idxs_pad.append(get_idxs(seq, to_ix))
    return idxs_pad


def prepare_sequence(seq, to_ix):
    # idxs = [to_ix[w] for w in seq]
    return torch.tensor(get_idxs(seq, to_ix), dtype=torch.long)


def prepare_sequence_batch(data, word_to_ix, tag_to_ix):
    seqs = [i[0] for i in data]
    tags = [i[1] for i in data]
    max_len = max([len(seq) for seq in seqs])
    seqs_pad = []
    tags_pad = []
    for seq, tag in zip(seqs, tags):
        seq_pad = seq + ['<PAD>'] * (max_len - len(seq))
        tag_pad = tag + ['<PAD>'] * (max_len - len(tag))
        seqs_pad.append(seq_pad)
        tags_pad.append(tag_pad)

    idxs_pad = torch.tensor(get_idxs_pad(seqs_pad, word_to_ix), dtype=torch.long)
    # idxs_pad = torch.tensor([[word_to_ix[w] for w in seq] for seq in seqs_pad], dtype=torch.long)
    tags_pad = torch.tensor([[tag_to_ix[t] for t in tag] for tag in tags_pad], dtype=torch.long)
    return idxs_pad, tags_pad


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_add(args):
    return torch.log(torch.sum(torch.exp(args), axis=0))


class BiLSTM_CRF_MODIFY_PARALLEL(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF_MODIFY_PARALLEL, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix  # 标签和index的对应的map
        self.tagset_size = len(tag_to_ix)  # 标签的数量

        self.word_embeds = nn.Embedding(vocab_size + 123, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,  # '//'表示整数除法
                            num_layers=1, bidirectional=True, batch_first=True)

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

    def _forward_alg_new_parallel(self, feats):
        # Do the forward algorithm to compute the partition function
        # 初始化一个feats.shape[0]x标签个数的矩阵，每一个都是-10000.0
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -10000.)  # .to('cuda')
        # 开始标签的得分置为最高
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[1]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            # gamar_r_l = torch.transpose(gamar_r_l,0,1)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)  # +1
            # t_r1_k = feats[:,feat_index,:].repeat(feats.shape[0],1,1).transpose(1, 2)
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            # forward_var_list.append(log_add(aa))
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]].repeat([feats.shape[0], 1])
        # terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).unsqueeze(dim=0)
        # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1).transpose(0,1)
        lstm_out, self.hidden = self.lstm(embeds)
        # lstm_out = lstm_out.view(embeds.shape[1], self.hidden_dim)
        lstm_out = lstm_out.squeeze()
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _get_lstm_features_parallel(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        # score = autograd.Variable(torch.Tensor([0])).to('cuda')
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags.view(-1)])

        # if len(tags)<2:
        #     print(tags)
        #     sys.exit(0)
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _score_sentence_parallel(self, feats, tags):
        # Gives the score of provided tag sequences
        # feats = feats.transpose(0,1)

        score = torch.zeros(tags.shape[0])  # .to('cuda')
        tags = torch.cat([torch.full([tags.shape[0], 1], self.tag_to_ix[START_TAG]).long(), tags], dim=1)
        for i in range(feats.shape[1]):
            feat = feats[:, i, :]
            score = score + \
                    self.transitions[tags[:, i + 1], tags[:, i]] + feat[range(feat.shape[0]), tags[:, i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:, -1]]
        return score

    def _viterbi_decode_new(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)  # .to('cuda')
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            gamar_r_l = torch.squeeze(gamar_r_l)
            next_tag_var = gamar_r_l + self.transitions
            # bptrs_t=torch.argmax(next_tag_var,dim=0)
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        # Transition to STOP_TAG
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood_parallel(self, sentences, tags):
        feats = self._get_lstm_features_parallel(sentences)
        forward_score = self._forward_alg_new_parallel(feats)
        gold_score = self._score_sentence_parallel(feats, tags)
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode_new(lstm_feats)
        return score, tag_seq


def predict_right_rate(datas, model, word_to_ix, tag_to_ix):
    right_count = 0
    word_size = 0
    with torch.no_grad():
        for data_t in datas:
            # gold_tags = data_t[1]
            precheck_sent = prepare_sequence(data_t[0], word_to_ix)
            precheck_tags = torch.Tensor([tag_to_ix[t] for t in data_t[1]])
            (tensor_num, pridict_tags) = model(precheck_sent)
            # b = torch.Tensor(pridict_tags)
            # a = precheck_tags.eq(torch.Tensor(pridict_tags)).numpy().sum()
            right_count += precheck_tags.eq(torch.Tensor(pridict_tags)).numpy().sum()
            word_size += len(data_t[1])
    return right_count / word_size
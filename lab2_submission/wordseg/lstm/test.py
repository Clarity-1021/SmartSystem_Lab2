def predict_right_rate(datas, model, word_to_ix):
    right_count = 0
    word_size = 0
    with torch.no_grad():
        for data_t in datas:
            gold_tags = data_t[1]
            precheck_sent = prepare_sequence(data_t[0], word_to_ix)
            # precheck_tags = torch.tensor([tag_to_ix[t] for t in data_t[1]], dtype=torch.long)
            pridict_tags = model(precheck_sent)
            right_count += pridict_tags.eq(gold_tags.view_as(pridict_tags)).numpy().sum()
            word_size += sum(data_t[1])
    return right_count / word_size
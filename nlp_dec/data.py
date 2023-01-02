import paddle

# 数据增强
def read_text_pair(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 2:
                continue
            yield {'text_a': data[0], 'text_b': data[1]}


# 无监督
def read_simcse_text(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip()
            yield {'text_a': data, 'text_b': data}


def gen_id2corpus(corpus_file):
    id2corpus = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            id2corpus[idx] = line.rstrip()
    return id2corpus


def convert_example_test(example, tokenizer, max_seq_length=512, pad_to_max_seq_len=False):
    result = []
    for key, text in example.items():
        encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length, pad_to_max_seq_len=pad_to_max_seq_len)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        result += [input_ids, token_type_ids]
    return result


def gen_text_file(similar_text_pair_file):
    text2similar_text = {}
    texts = []
    with open(similar_text_pair_file, 'r', encoding='utf-8') as f:
        for line in f:
            splited_line = line.rstrip().split("\t")
            if len(splited_line) != 2:
                continue

            text, similar_text = line.rstrip().split("\t")

            if not text or not similar_text:
                continue
            text2similar_text[text] = similar_text
            texts.append({"text": text})
    return texts, text2similar_text


def read_simcse_text(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip()
            # 无监督训练，text_a和text_b是一样的
            yield {'text_a': data, 'text_b': data}


def convert_example(example, tokenizer, max_seq_length=512, do_evalute=False):
    # 把文本转换成id的形式
    result = []

    for key, text in example.items():
        if 'label' in key:
            # do_evaluate
            result += [example['label']]
        else:
            # do_train
            encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length)
            input_ids = encoded_inputs["input_ids"]
            token_type_ids = encoded_inputs["token_type_ids"]
            result += [input_ids, token_type_ids]
    return result


def create_dataloader(dataset,
                      mode='train',
                      batch_size=64,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
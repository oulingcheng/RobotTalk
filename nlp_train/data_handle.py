import paddle
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

# 加载数据

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

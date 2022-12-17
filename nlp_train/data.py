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
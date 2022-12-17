import paddle
import numpy as np
import random

def word_repetition(input_ids, token_type_ids, dup_rate=0.32):
    """Word Reptition strategy."""
    input_ids = input_ids.numpy().tolist()
    token_type_ids = token_type_ids.numpy().tolist()

    batch_size, seq_len = len(input_ids), len(input_ids[0])
    repetitied_input_ids = []
    repetitied_token_type_ids = []
    rep_seq_len = seq_len
    for batch_id in range(batch_size):
        cur_input_id = input_ids[batch_id]
        actual_len = np.count_nonzero(cur_input_id)
        dup_word_index = []
        # If sequence length is less than 5, skip it
        if (actual_len > 5):
            dup_len = random.randint(a=0, b=max(2, int(dup_rate * actual_len)))
            # Skip cls and sep position
            dup_word_index = random.sample(
                list(range(1, actual_len - 1)), k=dup_len)

        r_input_id = []
        r_token_type_id = []
        for idx, word_id in enumerate(cur_input_id):
            # Insert duplicate word
            if idx in dup_word_index:
                r_input_id.append(word_id)
                r_token_type_id.append(token_type_ids[batch_id][idx])
            r_input_id.append(word_id)
            r_token_type_id.append(token_type_ids[batch_id][idx])
        after_dup_len = len(r_input_id)
        repetitied_input_ids.append(r_input_id)
        repetitied_token_type_ids.append(r_token_type_id)

        if after_dup_len > rep_seq_len:
            rep_seq_len = after_dup_len
    # Padding the data to the same length
    for batch_id in range(batch_size):
        after_dup_len = len(repetitied_input_ids[batch_id])
        pad_len = rep_seq_len - after_dup_len
        repetitied_input_ids[batch_id] += [0] * pad_len
        repetitied_token_type_ids[batch_id] += [0] * pad_len

    return paddle.to_tensor(repetitied_input_ids, dtype='int64'), paddle.to_tensor(
        repetitied_token_type_ids, dtype='int64')
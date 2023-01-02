# 加载飞桨的API
import paddle
# 加载PaddleNLP的API
import paddlenlp as ppnlp
# 加载系统的API
from nlp_dec.ann_util import build_index
from paddlenlp.data import Tuple, Pad
from paddlenlp.datasets import MapDataset
from nlp_dec.sim_cse import SimCSE
from nlp_dec.data import gen_id2corpus, read_text_pair, convert_example_test, create_dataloader
from configs import cfg
from functools import partial

# 在GPU环境下运行
paddle.set_device("gpu")
tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(cfg.model_name)
# 使用预训练模型
pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(cfg.model_name)
# 无监督+R-Drop，类似于多任务学习
simcse_model = SimCSE(
    pretrained_model,
    margin=cfg.margin,
    scale=cfg.scale,
    output_emb_size=cfg.output_emb_size)

# 加载模型
state_dict = paddle.load(cfg.train_model_name)
simcse_model.set_state_dict(state_dict)
simcse_model.eval()

# 构建问题对
QA_dict = {}
cnt = 0

for qa in read_text_pair(cfg.cx_qa):
    q = qa['text_a']
    a = qa['text_b']
    if q not in QA_dict:
        QA_dict[q] = []
    QA_dict[q].append(a)
    cnt += 1

id2corpus = gen_id2corpus(cfg.cx_q)
corpus_list = [{idx: text} for idx, text in id2corpus.items()]

trans_func_corpus = partial(
    convert_example_test,
    tokenizer=tokenizer,
    max_seq_length=cfg.max_seq_length)
batchify_fn_corpus = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_segment
): [data for data in fn(samples)]
corpus_ds = MapDataset(corpus_list)
corpus_data_loader = create_dataloader(
    corpus_ds,
    mode='predict',
    batch_size=cfg.batch_size,
    batchify_fn=batchify_fn_corpus,
    trans_fn=trans_func_corpus)

final_index = build_index(corpus_data_loader,
                          simcse_model,
                          output_emb_size=cfg.output_emb_size,
                          hnsw_max_elements=cfg.hnsw_max_elements,
                          hnsw_ef=cfg.hnsw_ef,
                          hnsw_m=cfg.hnsw_m)

# 从QA对中检索出答案
def simcse_FAQ(example):
    encoded_inputs = tokenizer(
        text=[example],
        max_seq_len=cfg.max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    input_ids = paddle.to_tensor(input_ids)
    token_type_ids = paddle.to_tensor(token_type_ids)
    cls_embedding = simcse_model.get_pooled_embedding(input_ids=input_ids, token_type_ids=token_type_ids)
    # print('提取特征:{}'.format(cls_embedding))
    recalled_idx, cosine_sims = final_index.knn_query(
        cls_embedding.numpy(), 10)
    # 检索到最接近的问题
    q_text = id2corpus[recalled_idx[0][0]]
    if q_text in QA_dict:
        answer = QA_dict[q_text]
        if len(answer) > 0:
            # 返回第一个
            return QA_dict[q_text][0]
        else:
            return ""
    else:
        return ""


if __name__ == '__main__':
    # 模型训练
    res = simcse_FAQ("康慧宝需要买多少")
    print("识别结果", res)

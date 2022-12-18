import os
import random
import time
# 加载飞桨的API
import paddle
# 加载PaddleNLP的API
import paddlenlp as ppnlp
# 加载系统的API
from functools import partial
from paddlenlp.data import Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from nlp_train.wr import word_repetition
from nlp_train.data import read_text_pair
from nlp_train.sim_cse import SimCSE
from nlp_train.data_handle import convert_example, create_dataloader
from configs import cfg

# 在GPU环境下运行
paddle.set_device("gpu")

# 加载数据集，数据增强：
train_ds = load_dataset(read_text_pair, data_path=cfg.train_set_file, lazy=False)
# 加载训练集， 无监督
# train_set_file='baoxian/train.csv'
# train_ds = load_dataset(read_simcse_text, data_path=train_set_file, lazy=False)

# 使用rocketqa开放领域的问答模型
tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(cfg.model_path)

# partial赋默认的值
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=cfg.max_seq_length)

# 对齐组装成小批次数据
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # query_input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # query_segment
    Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # title_input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # tilte_segment
): [data for data in fn(samples)]

# 构建训练的Dataloader
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=cfg.batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
# 展示一下输入的dataloader的数据
for idx, batch in enumerate(train_data_loader):
    if idx == 0:
        print(batch)
        break

num_training_steps = len(train_data_loader) * cfg.epochs

lr_scheduler = LinearDecayWithWarmup(cfg.learning_rate, num_training_steps, cfg.warmup_proportion)

pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(cfg.model_path)
# 无监督+R-Drop，类似于多任务学习
model = SimCSE(
    pretrained_model,
    margin=cfg.margin,
    scale=cfg.scale,
    output_emb_size=cfg.output_emb_size)

# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# AdamW优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=cfg.weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params)


def do_train(model, train_data_loader, **kwargs):
    save_dir = kwargs['save_dir']
    global_step = 0
    tic_train = time.time()
    for epoch in range(1, cfg.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch
            # sample的方式使用同义词语句和WR策略
            # 概率可以设置
            if (random.random() < 0.2):
                title_input_ids, title_token_type_ids = query_input_ids, query_token_type_ids
                query_input_ids, query_token_type_ids = word_repetition(query_input_ids, query_token_type_ids,
                                                                        cfg.dup_rate)
                title_input_ids, title_token_type_ids = word_repetition(title_input_ids, title_token_type_ids,
                                                                        cfg.dup_rate)
            # else:
            #     query_input_ids,query_token_type_ids=word_repetition(query_input_ids,query_token_type_ids,cfg.dup_rate)
            #     title_input_ids,title_token_type_ids=word_repetition(title_input_ids,title_token_type_ids,cfg.dup_rate)

            loss, kl_loss = model(
                query_input_ids=query_input_ids,
                title_input_ids=title_input_ids,
                query_token_type_ids=query_token_type_ids,
                title_token_type_ids=title_token_type_ids)
            # 加入R-Drop的损失优化，默认设置的是0.1，参数可以调
            loss = loss + kl_loss * 0.1
            # 每隔5个step打印日志
            global_step += 1
            if global_step % 5 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()
            # 反向梯度求导更新
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            # 每隔save_steps保存模型
            if global_step % cfg.save_steps == 0:
                save_path = os.path.join(save_dir, "model_%d" % global_step)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_param_path = os.path.join(save_path, 'model_state.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                tokenizer.save_pretrained(save_path)
    # 保存最后一个batch的模型
    save_path = os.path.join(save_dir, "model_%d" % global_step)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        save_param_path = os.path.join(save_path, 'model_state.pdparams')
        paddle.save(model.state_dict(), save_param_path)
        tokenizer.save_pretrained(save_path)


if __name__ == '__main__':
    # 模型训练
    do_train(model, train_data_loader, save_dir=cfg.save_dir)

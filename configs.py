import logging
import os

from easydict import EasyDict as edict

cfg = edict()
LOGGER = logging.getLogger('content_dialog')
cfg['data'] = './data/xxxx.yaml'
cfg['half'] = False
cfg['dnn'] = False
cfg['weights'] = './model/xxx.pt'
# cfg['model'] = './JuggingFace/model/bert-base-chinese'
cfg['model'] = './JuggingFace/model/ernie_3.0_x_base_ch_open'
cfg['gpu_id'] = 0

# 训练nlp
# cfg['scale'] = 20
# cfg['margin'] = 0.1
# cfg['output_emb_size'] = 256  # 可以根据实际情况进行设置
# 关键参数
scale = 20  # 推荐值: 10 ~ 30
margin = 0.1  # 推荐值: 0.0 ~ 0.2
epochs = 3

# 学习率设置
learning_rate = 5E-5
warmup_proportion = 0.0
weight_decay = 0.0
save_steps = 10

output_emb_size = 256  # 可以根据实际情况进行设置
max_seq_length = 64  # 序列的最大的长度，根据数据集的情况进行设置
batch_size = 64  # batch_size越大，效果会更好
dup_rate = 0.3  # 建议设置在0~0.3之间
save_dir = './checkpoints'
model_path = '../JuggingFace/model/ernie_3.0_x_base_ch_open'
train_set_file = './data/baoxianzhidao_filter.csv'

# 转换为绝对路径
base_dir = os.path.dirname(os.path.realpath(__file__))
cfg.data = os.path.join(base_dir, cfg.data)
cfg.weights = os.path.join(base_dir, cfg.weights)
cfg.model = os.path.join(base_dir, cfg.model)
#cfg.train_set_file = os.path.join(base_dir, 'nlp_train/data/baoxianzhidao_filter.csv')

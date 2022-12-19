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
cfg['model_path'] = 'rocketqa-zh-base-query-encoder'
#cfg['model_path'] = '/ai/python_code/RobotTalk/nlp_train/checkpoint/ERNIE_VIL2_BASE_ViT.pdparams'
cfg['train_set_file'] = '/ai/python_code/RobotTalk/nlp_train/data/baoxianzhidao_filter.csv'

# 训练nlp
# cfg['scale'] = 20
# cfg['margin'] = 0.1
# cfg['output_emb_size'] = 256  # 可以根据实际情况进行设置
# 关键参数
cfg['scale'] = 20  # 推荐值: 10 ~ 30
cfg['margin'] = 0.1  # 推荐值: 0.0 ~ 0.2
cfg['epochs'] = 30

# 学习率设置
cfg['learning_rate'] = 5E-5
cfg['warmup_proportion'] = 0.0
cfg['weight_decay'] = 0.0
cfg['ave_steps'] = 10

cfg['output_emb_size'] = 256  # 可以根据实际情况进行设置
cfg['max_seq_length'] = 64  # 序列的最大的长度，根据数据集的情况进行设置
cfg['batch_size'] = 8  # batch_size越大，效果会更好
cfg['dup_rate'] = 0.3  # 建议设置在0~0.3之间
cfg['save_dir'] = './models'
cfg['save_steps'] = 1000



# 转换为绝对路径
base_dir = os.path.dirname(os.path.realpath(__file__))
cfg.data = os.path.join(base_dir, cfg.data)
cfg.weights = os.path.join(base_dir, cfg.weights)
cfg.model = os.path.join(base_dir, cfg.model)
cfg.train_set_file = os.path.join(base_dir, cfg.train_set_file)
#cfg.model_path =  os.path.join(base_dir, cfg.model_path)

import os
import paddle
import paddlenlp as ppnlp
from nlp_dec.sim_cse import SimCSE
from configs import cfg

output_path='D:/ai/python_code/RobotTalk/nlp_dec/models'

pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(cfg.nlp_model_path)
# 无监督+R-Drop，类似于多任务学习
model = SimCSE(
    pretrained_model,
    margin=cfg.margin,
    scale=cfg.scale,
    output_emb_size=cfg.output_emb_size)

# 切换成eval模式，关闭dropout
model.eval()
# Convert to static graph with specific input description
model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64")  # segment_ids
        ])
# Save in static graph model.
save_path = os.path.join(output_path, "inference")
paddle.jit.save(model, save_path)
from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
# from satellite_dataset import MyDataset
from satellite_tiles_Yuzhou import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch
from torch.utils.data import random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import os
import time
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger

torch.cuda.empty_cache()

# image_dir = "/home/gridsan/qwang/satellite_images/zoom17/"
# data_dir = "/home/gridsan/qwang/JTL-transit_shared/deep_hybrid_model/data/"
print('Start loading data from Yuzhou train script')

image_dir = "fml/TwoCities_Images_Processed"
data_dir = ["fml/Two_cities_final_prompt.csv"]
hint_dir = "fml/TwoCities_Segmentation_RoadOnly_Results"
#hint_dir = "blue_shenhaowang/yuzhouchen1/fml/Streetview images/hint"
#hint_dir = "fml/Segmentation_Results"

# Configs
#resume_path = 'GenerativeUrbanDesign/models/control_sd15_ini.ckpt'
#resume_path = 'training_logs_new91/lightning_logs/version_66661581/checkpoints/epoch=2-step=21087.ckpt'
resume_path = 'training_logs_91_RoadOnly/lightning_logs/version_1437629/checkpoints/epoch=13-step=49210.ckpt'

batch_size = 8
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

#def custom_collate_fn(batch):
    #collated_batch = {}
    #collated_batch['jpg'] = torch.stack([item['jpg'] for item in batch])
    #collated_batch['hint'] = torch.stack([item['hint'] for item in batch])
    #collated_batch['txt'] = [item['txt'] for item in batch]  # 保留原始字符串列表
    #return collated_batch

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('GenerativeUrbanDesign/models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset(image_dir, data_dir, hint_dir)

# 随机划分 train / val
train_ratio = 0.9
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)  # 固定随机数种子
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

os.makedirs("training_logs_91_RoadOnly", exist_ok=True)
os.makedirs("training_logs_91_RoadOnly/checkpoints", exist_ok=True)

# 保存 validation 文件名列表
val_indices = val_dataset.indices  # random_split 后 val_dataset.indices 拿到的是原dataset的索引
val_image_ids = [dataset.image_list_id[idx] for idx in val_indices]  # ✅ 用 image_list_id，而不是 image_paths

val_save_path = "training_logs_91_RoadOnly/validation_files.csv"
pd.DataFrame(val_image_ids, columns=["filename"]).to_csv(val_save_path, index=False)

print(f"✅ Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
print(f"✅ Validation 图像列表已保存到: {val_save_path}")





#dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True) #,collate_fn=custom_collate_fn
train_loader = DataLoader(train_dataset, num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size, shuffle=False)


#logger = ImageLogger(batch_frequency=logger_freq)



# EarlyStopping: 如果连续5个验证集指标没有提升，就停止训练
early_stop_callback = EarlyStopping(
    monitor="val/loss",     # 监控 validation loss
    patience=5,             # 连续5次没有改善就停
    verbose=True,
    mode="min"              # val_loss越小越好
)

# ModelCheckpoint: 保存 validation loss 最好的模型
checkpoint_callback = ModelCheckpoint(
    monitor="val/loss",     # 监控 validation loss
    dirpath="training_logs_91_RoadOnly/checkpoints",  # 保存目录
    filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",  # 文件命名
    save_top_k=1,           # 只保留最好的一个
    mode="min"              # val_loss越小越好
)

class TrainLoggerCallback(Callback):
    def __init__(self, log_save_path="training_logs_91_RoadOnly/train_log.csv"):
        super().__init__()
        self.log_save_path = log_save_path
        self.start_time = None
        self.logs = []

        # 初始化保存文件
        if not os.path.exists(os.path.dirname(log_save_path)):
            os.makedirs(os.path.dirname(log_save_path))
        
        # 写入表头
        with open(self.log_save_path, 'w') as f:
            f.write("epoch,train_loss,val_loss,time_sec\n")
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        end_time = time.time()
        elapsed = end_time - self.start_time
        
        # 提取 train loss 和 val loss
        train_loss = trainer.callback_metrics.get('train/loss')
        val_loss = trainer.callback_metrics.get('val/loss')

        # 处理 None的情况（防止崩）
        train_loss = train_loss.item() if train_loss is not None else -1
        val_loss = val_loss.item() if val_loss is not None else -1

        # 保存一行
        with open(self.log_save_path, 'a') as f:
            f.write(f"{trainer.current_epoch},{train_loss:.6f},{val_loss:.6f},{elapsed:.2f}\n")

        print(f"✅ Logged Epoch {trainer.current_epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Time={elapsed:.2f}s")

train_logger_callback = TrainLoggerCallback(log_save_path="training_logs_91_RoadOnly/train_log.csv")



# 在创建 Trainer 前添加：
#logger_tb = TensorBoardLogger(
    #save_dir="training_logs_91_RoadOnly",
    #name="lightning_logs"
#)

dev_mode = False  # ✅ 改为 False 后即可正式训练
resume = True   #第一次跑使用False 
trainer = pl.Trainer(
    accelerator='gpu',devices=[0],   ###devices=[0],
    precision=16,
    fast_dev_run=dev_mode,
    callbacks=[train_logger_callback],  # ✅ 多加了两个回调！ logger, #early_stop_callback, checkpoint_callback,
    #logger=logger_tb,  # 添加这行
    #callbacks=[logger],
    default_root_dir="training_logs_91_RoadOnly",
    resume_from_checkpoint=resume_path if resume else None #第一次跑的时候不要添加这个
)

# Train!
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
#trainer.fit(model, dataloader)
#trainer.fit(model, dataloader, ckpt_path=resume_path)

# ✅ 加上总结输出
print("✅ Training finished.")

best_model_path = checkpoint_callback.best_model_path
best_val_loss = checkpoint_callback.best_model_score

print(f"✅ Best model saved at: {best_model_path}")
print(f"✅ Best Validation Loss: {best_val_loss:.6f}")

if hasattr(trainer, "early_stopping_callback") and trainer.early_stopping_callback is not None:
    if trainer.early_stopping_callback.stopped_epoch > 0:
        print(f"✅ Early stopped at epoch {trainer.early_stopping_callback.stopped_epoch}")
    else:
        print("✅ Training finished without early stopping.")
else:
    print("✅ No early stopping configured.")

print(f"✅ Total Epochs Completed: {trainer.current_epoch + 1}")
print(f"✅ Total Steps Completed: {trainer.global_step}")
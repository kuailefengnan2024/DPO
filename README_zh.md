# 简介

这是 [Diffusion-DPO](https://arxiv.org/abs/2311.12908) 的训练代码。该脚本改编自 [diffusers 库](https://github.com/huggingface/diffusers/tree/v0.20.0-release/examples/text_to_image)。

# 模型检查点

以下模型使用 StableDiffusion 模型初始化，并按照论文中的描述进行训练（可使用 [launchers/](launchers/) 脚本复现，假设有16个GPU，请相应地调整梯度累积）。

[StableDiffusion1.5](https://huggingface.co/mhdang/dpo-sd1.5-text2image-v1)

[StableDiffusion-XL-1.0](https://huggingface.co/mhdang/dpo-sdxl-text2image-v1?text=Test)

使用此 [notebook](quick_samples.ipynb) 比较生成结果。它还有一个使用 PickScore 进行自动定量评估的示例。

# 安装

`pip install -r requirements.txt`

# 结构

- `launchers/` 是运行 SD1.5 或 SDXL 训练的示例
- `utils/` 包含用于评估或 AI 反馈的评分模型 (PickScore, HPS, Aesthetics, CLIP)
- `quick_samples.ipynb` 是预训练模型与基线模型的可视化对比
- `requirements.txt` 基本的 pip 依赖
- `train.py` 主脚本，这个文件比较大，超过1000行，训练循环在此提交的大约第1000行开始（`ctrl-F` 搜索 "for epoch"）。
- `upload_model_to_hub.py` 将模型检查点上传到 HF（一个简单的实用工具，当前值为占位符）

# 运行训练

SD1.5 启动示例

```bash
# from launchers/sd15.sh
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="yuvalkirstain/pickapic_v2"

# 有效批量大小 (Effective BS) 将是 (N_GPU * train_batch_size * gradient_accumulation_steps)
# 论文中使用了 2048。训练大约需要 24 小时 / 2000 步

accelerate launch --mixed_precision="fp16"  train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/export/share/datasets/vision_language/pick_a_pic_v2/" \
  --checkpointing_steps 500 \
  --beta_dpo 5000 \
   --output_dir="tmp-sd15"
```

## 重要参数

### 通用

- `--pretrained_model_name_or_path` 从哪个模型进行训练/初始化
- `--output_dir` 保存/记录日志的位置
- `--seed` 训练种子（默认未设置）
- `--sdxl` 运行 SDXL 训练
- `--sft` 运行 SFT 而不是 DPO

### DPO

- `--beta_dpo` DPO 的 KL 散度参数 beta
- `--choice_model` 用于 AI 反馈的模型 (Aesthetics, CLIP, PickScore, HPS)

### 优化器/学习率

- `--max_train_steps` 进行多少训练步数
- `--gradient_accumulation_steps`
- `--train_batch_size` 实际的批量大小请参见脚本中的注释
- `--checkpointing_steps` 保存模型的频率
  
- `--gradient_checkpointing` 对于 SDXL 自动开启

- `--learning_rate`
- `--scale_lr` 发现这个参数很有用，但在代码中不是默认设置
- `--lr_scheduler` 学习率预热/衰减的类型。默认为线性预热至常数
- `--lr_warmup_steps` 调度器的预热步数
- `--use_adafactor` 使用 Adafactor 替代 Adam（内存占用更低，SDXL 默认使用）

### 数据
- `--dataset_name` 如果你想从 Pick-a-Pic 切换
- `--cache_dir` 数据集在本地缓存的位置 **(用户需要根据自己的文件系统更改此项)**
- `--resolution` 非 SDXL 默认为 512，SDXL 默认为 1024。
- `--random_crop` 和 `--no_hflip` 更改数据增强方式
- `--dataloader_num_workers` 数据加载器的总工作进程数


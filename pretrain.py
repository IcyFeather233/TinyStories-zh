# -*- coding: utf-8 -*-
"""
## **1. 安装依赖**
"""

# !pip install -q datasets==2.18.0
# !pip install -U accelerate

"""## **2. 模型初始化**"""

import os
import torch

os.environ['WANDB_DISABLED'] = 'true'                       # 禁用 wandb，也可以不用这一条
device = 'cuda' if torch.cuda.is_available() else 'cpu'     # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

tokenizer = AutoTokenizer.from_pretrained('./my_model/hf_tokenizer/')

hidden_size = 512
intermediate_size = (int(hidden_size * 8/3 / 128) + 1) * 128
vocab_size = 40960

config = AutoConfig.for_model(
    model_type="llama",
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_attention_heads=16,
    num_hidden_layers=4,
    num_key_value_heads=8,
    vocab_size=vocab_size
)
print(config)

import torch

model = AutoModelForCausalLM.from_config(
    config,
    torch_dtype=torch.float32
).to(device)
print(model)

# 训练的时候要设置 left padding
tokenizer.padding_side='left'
print(tokenizer)


# 打印模型的每一层及其参数大小
def print_model_parameters(model):
    print("Layer Name & Parameters")
    print("----------------------------")
    total_params = 0
    for name, parameter in model.named_parameters():
        param_size = parameter.size()
        param_count = torch.prod(torch.tensor(param_size)).item()
        total_params += param_count
        print(f"{name:50} | Size: {str(param_size):30} | Count: {str(param_count):20}")
    print("----------------------------")
    print(f"Total Parameters: {total_params} ({total_params / 1000000:.1f} M)")

print_model_parameters(model)

def inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str = "从前，",
    max_new_tokens: int = 16
):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        return_token_type_ids=False
    ).to(device)
    
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.8
    )
    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    # print(outputs)
    print(generated_text)

print('test inference before training:')
inference(model, tokenizer)

# Kaiming 初始化
def kaiming_initialization(model):
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            torch.nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='leaky_relu')
        elif 'bias' in name:
            # 一般偏置项可以初始化为0
            torch.nn.init.constant_(param, 0)

kaiming_initialization(model)
inference(model, tokenizer)

"""## **3. 数据集**"""

from datasets import load_dataset

# 加载整个训练数据集
full_train_dataset = load_dataset('/maindata/data/user/mengzhuo.chen/datasets/TinyStories-Zh-1M', split='train')

# 计算分割点
split_index = int(0.9 * len(full_train_dataset))

# 按9:1分割数据集
ds_train = full_train_dataset.select(range(split_index))
ds_val = full_train_dataset.select(range(split_index, len(full_train_dataset)))

print(ds_train)
print(ds_val)

# 查看一下数据示例
print(ds_train[:2])

from typing import Dict, List

def process_func(examples: Dict[str, List]):
    max_token = 2048

    encoded_texts = tokenizer(examples['text'], add_special_tokens=False)
    input_ids_list = encoded_texts['input_ids']

    new_input_ids_list, new_attn_mask_list = [], []
    for input_ids in input_ids_list:
        temp = input_ids[-max_token+1:] + [tokenizer.eos_token_id]
        new_input_ids_list.append(temp)
        new_attn_mask_list.append([1] * len(temp))
    return {
        "input_ids": new_input_ids_list,
        "attention_mask": new_attn_mask_list
    }

ds_train = ds_train.shuffle()

PROC_NUM = 32

ds_train = ds_train.map(
    process_func,
    batched=True,
    num_proc=PROC_NUM,
    remove_columns=ds_train.column_names,
    desc='Running tokenizer on train_set: '
)
ds_val = ds_val.map(
    process_func,
    batched=True,
    num_proc=PROC_NUM,
    remove_columns=ds_val.column_names,
    desc='Running tokenizer on val_set: '
)

print(ds_train)
print(ds_val)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

"""## **4. 训练**"""

from transformers import TrainingArguments

BATCH_SIZE = 96

training_args = TrainingArguments(
    output_dir='saves',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    eval_steps=1000,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    lr_scheduler_type='cosine',
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    logging_steps=50,
    report_to=None,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    seed=3407
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

print('test inference after training:')
inference(
    model,
    tokenizer,
    "从前，",
    max_new_tokens=100
)

"""## **5. 保存模型**"""

# 保存到本地
model.save_pretrained('my_model')


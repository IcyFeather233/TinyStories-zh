import os
import torch

os.environ['WANDB_DISABLED'] = 'true'                       # 禁用 wandb，也可以不用这一条
device = 'cuda' if torch.cuda.is_available() else 'cpu'     # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

tokenizer = AutoTokenizer.from_pretrained('./my_model/hf_tokenizer/')

model = AutoModelForCausalLM.from_pretrained('./my_model/').to(device)
print(model)

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

# left padding

tokenizer.padding_side='left'
print(tokenizer)

print('test left padding:')
inference(
    model,
    tokenizer,
    "从前，",
    max_new_tokens=100
)

# right padding

tokenizer.padding_side='right'
print(tokenizer)

print('test right padding:')
inference(
    model,
    tokenizer,
    "从前，",
    max_new_tokens=100
)

# left padding 和 right padding 在推理的时候基本没有区别










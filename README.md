# TinyStories-zh

## 背景

TinyStories 原工作探索的问题是语言模型（LM）在文本连贯性上的表现。像早期的一些语言模型如 GPT-2，即使在一些 Common Craw 这样的语料库上大量预训练后，也很难生成长的、连贯的文本。比如前几年有一种 AI 玩具类型是做文本续写，例如彩云小梦，可以写写作文、小说什么的，如果大家玩过就知道效果其实一言难尽，和今天的大模型完全没法比，其实这就是 GPT-2 level 的续写能力。

作者就在想，会不会是因为训练的语料库太多、太宽泛，需要学习各种语法元素、词汇、知识、推理等等，才导致小语言模型（SLM）没法有一个很好的表现。作者决定专注于一个任务——短篇故事续写，来探索一下 LM 的性能边界。

作者用 GPT-4 和 GPT-3.5 构建了一个英文短篇小说数据集 TinyStories，将内容限制在三四岁儿童也能轻松理解的程度，并且使用不同的关键词来让故事的主题足够丰富。此外，他们还加入了额外的关键词，来控制故事有更曲折的走向、不同的结局等等。

作者用的模型基座架构是 GPT Neo，词表大小约为 50k，并且他们尝试了不同的模型参数，调整了隐藏层维度（hidden_size）、隐藏层数（num_hidden_layers）等，来探索不同参数对于模型性能的影响。

此工作是对 TinyStories 工作的复现。

## 数据集

[52AI/TinyStoriesZh](https://huggingface.co/datasets/52AI/TinyStoriesZh) 进行了翻译，本项目将该数据集处理成了和原始的 [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 一样的 jsonl 格式，上传为 [RobinChen2001/TinyStories-Zh-1M](https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-1M) 和 [RobinChen2001/TinyStories-Zh-2M](https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-2M) 数据集

故事的平均长度: 251.81

## Tokenizer

项目使用的是用训练数据集训练出来的 tokenizer，vocab_size 选择的是 40960

注意 `train_tokenizer.py` 里面我是直接对 jsonl 文件进行的处理，如果是从 huggingface 上下载的数据集，下载下来是 parquet 文件，需要稍微改动一下代码

## 实验

使用下面的配置：

```
LlamaConfig {
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 1408,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 16,
  "num_hidden_layers": 4,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "transformers_version": "4.40.2",
  "use_cache": true,
  "vocab_size": 40960
}
```

用 1M 的数据跑 3 epoch

```
{'train_runtime': 4787.9214, 'train_samples_per_second': 573.506, 'train_steps_per_second': 5.974, 'train_loss': 3.9153821066696524, 'epoch': 3.0}
```

测试代码：

```
inference(
    model,
    tokenizer,
    "从前，",
    max_new_tokens=100
)
```

结果：

```
从前，有一个小男孩，名叫蒂米。提米非常听话，总是听爸爸妈妈的话。有一天，蒂米的妈妈让他去趟厕所。
提米去了厕所，但马桶坏了。他开始挣扎，试图修理它。他试图解决这个问题，但他的拇指太疼了。
妈妈看到他哭了，就问他怎么了。蒂米告诉他妈妈他的膝盖受伤了。他妈妈向他解释说，有时很难做某事，但保持不动。
蒂米吸取了教训，承诺以后会更加小心。从那天起，蒂米总是听妈妈的话，以免再受伤。仍然很严肃，但可以解决任何问题。的水龙头和马桶保持整洁，蒂米也不再痛苦了。不再乱，蒂米学到了关于
```

分析：
1. 已经有了初步讲故事的能力，但是句子还是不通顺
2. 这种现象可能是因为 TinyStories 数据集机器翻译质量不高，例如“蒂米”“提米”混用，可能是翻译的时候同一个名字翻译成了两种形式造成的
3. 模型大小/数据量可能还不够

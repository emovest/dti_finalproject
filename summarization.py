## Summarization ##
## Providing Summarization of the abstracts of User's liken Papers ##
## Using PRIMERA Model ##

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch

# 加载 PRIMERA 模型
model_name = "allenai/PRIMERA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_with_primera(papers_df, text_column="original_abstract", max_input_tokens=4096, max_output_tokens=768):
    # 拼接所有摘要，并加上特殊分隔符
    abstracts = papers_df[text_column].tolist()
    full_input = " </s> ".join(abstracts)  # PRIMERA 使用 </s> 作为段落分隔

    # 编码输入文本
    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        padding=True
    )

    # 生成摘要
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_output_tokens,
        min_length=250,
        num_beams=4,
        length_penalty=0.8,
        early_stopping=False,
    )

    # 解码输出
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
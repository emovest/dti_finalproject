## Summarization ##
## Providing Summarization of the abstracts of User's liken Papers ##
## Using BART Model ##

from transformers import pipeline
from transformers import pipeline, AutoTokenizer

# 初始化
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def summarize_papers_with_bart(papers_df, text_column="original_abstract", max_tokens=400, min_tokens=200):
    abstracts = papers_df[text_column].tolist()
    full_text = " ".join(abstracts)

    # Tokenize 并自动截断到 1024 token（BART 上限）
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
    decoded_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    # 生成摘要
    summary = summarizer(
        decoded_text,
        max_length=max_tokens,
        min_length=min_tokens,
        do_sample=False
    )[0]["summary_text"]

    return summary
## Paper Recommender Part 3 ##
## Recommender Additional Five Papers Based on User's Frist Liked Paper ## 
## Using content-based, Cosine Similarity Algorithm ##

def recommend_more_from_liked_paper(liked_paper_text, label, top_k=5):
    # 找到该 label 对应的子集
    df = pd.read_csv("cleaned_papers_with_id.csv")
    mask = df["label"] == label
    label_subset_df = df[mask].reset_index(drop=True)
    label_subset_embeddings = embeddings[mask.values]

    if len(label_subset_df) <= 1:
        print("❌ Not enough papers in this label to recommend more.")
        return None

    # 嵌入喜欢的那篇论文（title+abstract）
    liked_embedding = embedding_model.encode(liked_paper_text)

    # 计算相似度
    cosine_scores = util.cos_sim(liked_embedding, label_subset_embeddings)[0]

    # 排除自己（相似度 = 1.0 的那篇），选出 top_k+1 再排除 index 0（假设相同）
    top_indices = cosine_scores.argsort(descending=True)[1:top_k+1]
    top_papers = label_subset_df.iloc[top_indices]

    for idx, row in enumerate(top_papers.itertuples(), 1):
        print(f"\n📄 Paper You Might Also Like NO.{idx}:")
        print("Paper Title:", row.original_title)
        print("Abstract:")
        print(row.original_abstract)

    return top_papers

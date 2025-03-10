import sqlite3
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pdf_reader import process_pdfs_in_directory

import os
# if os.path.exists('embeddings.db'):
#     os.remove('embeddings.db')

def get_my_model():
    """
    加载模型函数

    该函数加载两个预训练模型和一个tokenizer，用于后续的文本嵌入和重排序任务
    """
    bgem3_model_path = '../bge-m3'
    bge_rerank_model_path = '../bge-rerank'

    model1 = SentenceTransformer(bgem3_model_path)

    model2 = AutoModelForSequenceClassification.from_pretrained(bge_rerank_model_path)
    model2.eval()
    tokenizer = AutoTokenizer.from_pretrained(bge_rerank_model_path)
    return model1, model2, tokenizer


def get_pdf_chunks_embedding(pdf_file_dir: str, model, chunk_size=1000, chunk_overlap=200):
    """
    获取PDF块嵌入函数

    该函数将目录中的所有PDF文件处理成文本块，并计算每个文本块的嵌入向量
    参数:
        pdf_file_dir: PDF文件所在目录
        model: 用于生成文本嵌入的模型
        chunk_size: 每个文本块的大小
        chunk_overlap: 文本块之间的重叠大小
    返回:
        chunks_embeddings: 所有文本块的嵌入向量
        all_chunks: 所有的文本块
        all_chunks_idx: 文本块的索引
    """
    # chunks = splitter(pdf_file_path)
    all_chunks, all_chunks_idx = process_pdfs_in_directory(pdf_file_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks_embeddings = torch.from_numpy(model.encode(all_chunks, normalize_embeddings=True))
    return chunks_embeddings, all_chunks, all_chunks_idx


def get_n_maximum_similarity(query_embedding, chunks_embeddings, n=5):
    """
    获取最大相似度的文本块索引和相似度值

    参数:
        query_embedding: 查询文本的嵌入向量
        chunks_embeddings: 所有文本块的嵌入向量
        n: 返回的文本块数量
    返回:
        top_n_index: 最大相似度的文本块索引
        top_n_similarity: 最大相似度的值
    """
    similarity_query = query_embedding @ chunks_embeddings.T

    top_n_index = torch.topk(similarity_query, k=n, dim=1).indices
    top_n_similarity = torch.topk(similarity_query, k=n, dim=1).values
    return top_n_index, top_n_similarity


def get_rerank_score(query_list: list, top_n_index: torch.Tensor, chunks, rerank_model, rerank_tokenizer):
    """
    获取重排序分数函数

    该函数使用重排序模型对查询文本和候选文本块进行重排序，以获得更准确的相关性分数
    参数:
        query_list: 查询文本列表
        top_n_index: 候选文本块的索引
        chunks: 所有的文本块
        rerank_model: 用于重排序的模型
        rerank_tokenizer: 用于重排序的tokenizer
    返回:
        scores: 重排序后的相关性分数
    """
    num_queries = len(query_list)
    num_chunks = top_n_index.shape[1]
    pairs = []

    for i in range(num_queries):
        pairs_raw = []
        for j in range(num_chunks):
            pair = [query_list[i], chunks[top_n_index[i][j]]]
            pairs_raw.append(pair)
        pairs.append(pairs_raw)

    all_pairs = [pair for sublist in pairs for pair in sublist]
    with torch.no_grad():
        inputs = rerank_tokenizer(all_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        logits = rerank_model(**inputs, return_dict=True).logits
        scores = logits.view(num_queries, num_chunks).float()
    return scores


def create_database():
    """
    创建一个名为embeddings.db的SQLite数据库，并在其中创建一个名为embeddings的表。
    如果embeddings表已经存在，则不创建。

    表结构：
    - id: 整数类型，主键，自动递增。
    - chunk: 文本类型，用于存储数据块。
    - embedding: 二进制大对象类型，用于存储嵌入数据。
    - idx: 整数类型，用于存储数据块的原始索引。
    """
    # 连接到SQLite数据库，如果数据库不存在则会自动创建。
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  chunk TEXT,
                  embedding BLOB,
                  idx INTEGER)''')
    conn.commit()
    conn.close()


def insert_into_database(chunks, embeddings, all_chunks_idx):
    """
    将分块文本、对应的嵌入向量和原始索引插入数据库。

    参数:
    chunks (list): 分块后的文本列表。
    embeddings (list): 文本的嵌入向量列表，每个嵌入向量对应一个分块文本。
    all_chunks_idx (list): 文本块的原始索引列表。
    """
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    for i in range(len(chunks)):
        c.execute("INSERT INTO embeddings (chunk, embedding, idx) VALUES (?, ?, ?)",
                  (chunks[i], embeddings[i].tobytes(), all_chunks_idx[i]))
    conn.commit()
    conn.close()


def read_embeddings():
    """
    从SQLite数据库中读取预计算的文本嵌入和原始索引。

    此函数连接到名为'embeddings.db'的SQLite数据库，从中选择所有嵌入数据和原始索引。
    它将文本块、对应的嵌入向量和原始索引分别存储在三个列表中，并返回这三个列表。

    Returns:
        chunks: 包含所有文本块的列表。
        embeddings: 包含所有对应嵌入向量的列表。
        all_chunks_idx: 包含所有文本块原始索引的列表。
    """
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute("SELECT chunk, embedding, idx FROM embeddings")
    rows = c.fetchall()
    chunks = [row[0] for row in rows]
    embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in rows]
    all_chunks_idx = [row[2] for row in rows]
    conn.close()
    return chunks, embeddings, all_chunks_idx


def main_function():
    bgem3_model, rerank_model, rerank_tokenizer = get_my_model()
    pdf_file_dir = './pdf_dir'
    chunks_embeddings, chunks, all_chunks_idx = get_pdf_chunks_embedding(pdf_file_dir, bgem3_model, 2000, 400)

    create_database()
    insert_into_database(chunks, chunks_embeddings.numpy(), all_chunks_idx)


def main():
    """
    主函数

    该函数协调上述所有函数，实现从PDF处理、文本嵌入、相似度计算到重排序的全过程
    """
    bgem3_model, rerank_model, rerank_tokenizer = get_my_model()

    if not os.path.exists('embeddings.db'):
        pdf_file_dir = './pdf_dir'
        chunks_embeddings, chunks, all_chunks_idx = get_pdf_chunks_embedding(pdf_file_dir, bgem3_model, 2000, 400)

        create_database()
        insert_into_database(chunks, chunks_embeddings.numpy(), all_chunks_idx)

    db_chunks, db_embeddings, all_chunks_idx = read_embeddings()
    db_embeddings = torch.from_numpy(np.stack(db_embeddings))
    print("chunk_embeddings shape: ", db_embeddings.shape)

    query_list = ["什么是气溶胶光散射吸湿性?", "SMPS仪器是什么?", "FIMS仪器是什么?"]
    query_embedding = torch.from_numpy(bgem3_model.encode(query_list, normalize_embeddings=True))

    top_n_index, top_n_similarity = get_n_maximum_similarity(query_embedding, db_embeddings)
    print("top_n_index: ", top_n_index, "\n", "top_n_similarity: ", top_n_similarity)

    rerank_scores = get_rerank_score(query_list, top_n_index, db_chunks, rerank_model, rerank_tokenizer)
    print("rerank_score: ", rerank_scores)

    for i in range(len(query_list)):
        max_index = torch.argmax(rerank_scores[i])
        print(all_chunks_idx[top_n_index[i][max_index]], db_chunks[top_n_index[i][max_index]])


if __name__ == '__main__':
    main()

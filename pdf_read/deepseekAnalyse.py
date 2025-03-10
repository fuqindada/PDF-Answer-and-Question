import os
import numpy as np
import torch
from langchain_deepseek import ChatDeepSeek
from query_search import get_my_model, read_embeddings, get_n_maximum_similarity, get_rerank_score

def get_model():
    os.environ["DEEPSEEK_API_KEY"] = "sk-58eddbec03f341afb094410c381d1e4f"
    model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    return model


def main():
    if not os.path.exists('embeddings.db'):
        from query_search import main_function
        main_function()

    bgem3_model, rerank_model, rerank_tokenizer = get_my_model()
    query_list = ["什么是气溶胶光散射吸湿性?", "SMPS仪器是什么?", "FIMS仪器是什么?"]
    query_embedding = torch.from_numpy(bgem3_model.encode(query_list, normalize_embeddings=True))
    db_chunks, db_embeddings, all_chunks_idx = read_embeddings()
    db_embeddings = torch.from_numpy(np.stack(db_embeddings))

    top_n_index, _ = get_n_maximum_similarity(query_embedding, db_embeddings)

    model = get_model()

    for i in range(len(query_list)):
        top_n_chunks = [db_chunks[j] for j in top_n_index[i]]
        top_n_chunks_str = "\n\n".join(top_n_chunks)
        messages = [
            (
                "system",
                "您是一个有用的助手，根据提出的问题，将所给的内容进行提炼总结。",
            ),
            ("human", "问题：" + query_list[i] + "\n\n" + "内容：" + top_n_chunks_str),
        ]
        ai_msg = model.invoke(messages)
        ai_msg_ = ai_msg.content.replace('\n\n', '\n')
        print(query_list[i], "\n答: ", ai_msg_)
        print("\n")


if __name__ == "__main__":
    main()

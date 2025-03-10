## README.md

### 项目概述

本项目旨在实现从PDF文件中查询相关文本，并通过预训练的深度学习模型进行文本嵌入和重排序，以实现高效的查询和检索功能。项目包含两个主要模块：

1. **query_search.py**：负责加载模型、处理PDF文件、计算文本嵌入、相似度匹配以及重排序。
2. **pdf_reader.py**：负责读取PDF文件内容、清理文本、分割文本块。

### 文件结构

```
.
├── bge_m3/*
├── bge-rerank/*
├── pdf_reader
        ├── pdf_dir/
        ├── query_search.py
        └── pdf_reader.py
├── demo.py
└── README.md
```


### 功能说明

### bge-m3/
- **下放预训练的embedding模型**：
- 下载链接：https://platform.virtaicloud.com/gemini_web/workspace/space/ixd7dxy5ecw2/model/detail/499462114227462144

### bge-rerank/
- **下放预训练的rerank模型**：
- 下载链接：https://platform.virtaicloud.com/gemini_web/workspace/space/ixd7dxy5ecw2/model/detail/460415399646380032

#### query_search.py

- **加载模型 (`get_my_model`)**：
  - 加载两个预训练模型（`SentenceTransformer` 和 `AutoModelForSequenceClassification`）及对应的 `tokenizer`，用于后续的文本嵌入和重排序任务。

- **获取PDF块嵌入 (`get_pdf_chunks_embedding`)**：
  - 将指定目录中的所有PDF文件处理成文本块，并计算每个文本块的嵌入向量。
  - 参数:
    - `pdf_file_dir`: PDF文件所在目录
    - `model`: 用于生成文本嵌入的模型
    - `chunk_size`: 每个文本块的大小，默认为1000
    - `chunk_overlap`: 文本块之间的重叠大小，默认为200
  - 返回:
    - `chunks_embeddings`: 所有文本块的嵌入向量
    - `all_chunks`: 所有的文本块
    - `all_chunks_idx`: 文本块的索引

- **获取最大相似度的文本块索引和相似度值 (`get_n_maximum_similarity`)**：
  - 计算查询文本与所有文本块之间的相似度，并返回前N个最相似的文本块及其相似度值。
  - 参数:
    - `query_embedding`: 查询文本的嵌入向量
    - `chunks_embeddings`: 所有文本块的嵌入向量
    - `n`: 返回的文本块数量，默认为5
  - 返回:
    - `top_n_index`: 最大相似度的文本块索引
    - `top_n_similarity`: 最大相似度的值

- **获取重排序分数 (`get_rerank_score`)**：
  - 使用重排序模型对查询文本和候选文本块进行重排序，以获得更准确的相关性分数。
  - 参数:
    - `query_list`: 查询文本列表
    - `top_n_index`: 候选文本块的索引
    - `chunks`: 所有的文本块
    - `rerank_model`: 用于重排序的模型
    - `rerank_tokenizer`: 用于重排序的tokenizer
  - 返回:
    - `scores`: 重排序后的相关性分数

- **主函数 (`main`)**：
  - 协调上述所有函数，实现从PDF处理、文本嵌入、相似度计算到重排序的全过程。
  - query_list 为查询文本列表
  - 输出打印的文本块相似度值，以及重排序分数、重排序分数最大的文本块索引。

#### pdf_reader.py

- **清理文本 (`clean_text`)**：
  - 清理文本，包括去除多余的空行和空格。
  - 参数:
    - `text`: 需要清理的文本
  - 返回:
    - 清理后的文本

- **读取PDF文件内容 (`read_pdf_with_pdfplumber`)**：
  - 使用 `pdfplumber` 库读取PDF文件内容。
  - 参数:
    - `pdf_path`: PDF文件路径
  - 返回:
    - PDF文件的文本内容

- **递归字符文本分割器 (`RecursiveCharacterTextSplitter`)**：
  - 递归地将文本分割成指定大小的块，并处理文本块之间的重叠。
  - 参数:
    - `chunk_size`: 每个文本块的大小，默认为1000
    - `chunk_overlap`: 文本块之间的重叠大小，默认为200
    - `separators`: 文本分割符列表，默认为["\n\n", "\n", ".", "!", "?", " "]

- **分割PDF文件文本 (`splitter`)**：
  - 分割PDF文件文本。
  - 参数:
    - `pdf_file_path`: PDF文件路径
    - `chunk_size`: 每个文本块的大小，默认为1000
    - `chunk_overlap`: 文本块之间的重叠大小，默认为200
  - 返回:
    - 分割后的文本块列表

- **新的分割PDF文件文本的方法 (`new_splitter`)**：
  - 使用 `langchain_text_splitters` 库中的 `RecursiveCharacterTextSplitter` 进行文本分割。
  - 参数:
    - `pdf_file_path`: PDF文件路径
    - `chunk_size`: 每个文本块的大小，默认为1000
    - `chunk_overlap`: 文本块之间的重叠大小，默认为200
  - 返回:
    - 分割后的文本块列表

- **处理目录中的所有PDF文件 (`process_pdfs_in_directory`)**：
  - 处理指定目录中的所有PDF文件，并返回所有分割后的文本块及其索引。
  - 参数:
    - `directory`: 目录路径
    - `chunk_size`: 每个文本块的大小，默认为1000
    - `chunk_overlap`: 文本块之间的重叠大小，默认为200
  - 返回:
    - 所有分割后的文本块和对应的索引

### 依赖项

请确保安装以下依赖项：

```bash
pip install torch sentence-transformers transformers pdfplumber 
```


### 使用方法

1. 将PDF文件放入 `./pdf_dir` 目录。
2. 运行 `query_search.py` 文件：

```bash
python query_search.py
```


### 注意事项

- 确保PDF文件路径正确。
- 根据实际需求调整`query_search.py`中`query_list`、 `chunk_size` 和 `chunk_overlap` 等参数。
- 如果使用 `langchain_text_splitters` 库，请确保其已正确安装并配置，如不使用，将pdf_reader.py的`process_pdfs_in_directory`函数中`new_splitter`改为`splitter`，从而使用代码内自行实现的 `RecursiveCharacterTextSplitter`。

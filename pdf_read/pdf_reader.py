import pdfplumber
import re
import os


def clean_text(text):
    """
    清理文本，包括去除多余的空行和空格。

    参数:
    text (str): 需要清理的文本。

    返回:
    str: 清理后的文本。
    """
    # 去除多余的空行和空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空格
    text = text.strip()
    return text


def read_pdf_with_pdfplumber(pdf_path):
    """
    使用pdfplumber读取PDF文件内容。

    参数:
    pdf_path (str): PDF文件路径。

    返回:
    str: PDF文件的文本内容。
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    cleaned_text = clean_text(text)
    return cleaned_text


class RecursiveCharacterTextSplitter:
    """
    递归字符文本分割器。

    参数:
    chunk_size (int): 每个文本块的大小。
    chunk_overlap (int): 文本块之间的重叠大小。
    separators (list): 文本分割符列表。
    """

    def __init__(self, chunk_size=1000, chunk_overlap=200, separators=None):
        if separators is None:
            separators = ["\n\n", "\n", ".", "!", "?", " "]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text):
        """
        分割文本。

        参数:
        text (str): 需要分割的文本。

        返回:
        list: 分割后的文本块列表。
        """
        return self._split_text(text, self.separators)

    def _split_text(self, text, separators):
        """
        内部分割文本的逻辑。

        参数:
        text (str): 需要分割的文本。
        separators (list): 分割符列表。

        返回:
        list: 分割后的文本块列表。
        """
        if len(text) <= self.chunk_size:
            return [text]

        for separator in separators:
            splits = text.split(separator)
            if len(splits) > 1:
                chunks = []
                current_chunk = ""
                for split in splits:
                    if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                        current_chunk += split + separator
                    else:
                        if len(current_chunk) > 0:
                            chunks.append(current_chunk.strip())
                            current_chunk = split + separator
                if len(current_chunk) > 0:
                    chunks.append(current_chunk.strip())

                # Handle overlap
                overlapped_chunks = []
                for i in range(len(chunks) - 1):
                    overlapped_chunks.append(chunks[i])
                    if i < len(chunks) - 1:
                        overlap = chunks[i][-self.chunk_overlap:]
                        next_chunk = overlap + chunks[i + 1]
                        overlapped_chunks.append(next_chunk)

                if len(chunks) > 0:
                    overlapped_chunks.append(chunks[-1])

                return overlapped_chunks

        # Fallback to splitting by character if no separators found
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]


def splitter(pdf_file_path, chunk_size=1000, chunk_overlap=200):
    """
    分割PDF文件文本。

    参数:
    pdf_file_path (str): PDF文件路径。
    chunk_size (int): 每个文本块的大小。
    chunk_overlap (int): 文本块之间的重叠大小。

    返回:
    list: 分割后的文本块列表。
    """
    content = read_pdf_with_pdfplumber(pdf_file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(content)
    return chunks


def new_splitter(pdf_file_path, chunk_size=1000, chunk_overlap=200):
    """
    新的分割PDF文件文本的方法。

    参数:
    pdf_file_path (str): PDF文件路径。
    chunk_size (int): 每个文本块的大小。
    chunk_overlap (int): 文本块之间的重叠大小。

    返回:
    list: 分割后的文本块列表。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    content = read_pdf_with_pdfplumber(pdf_file_path)
    chunks = text_splitter.split_text(content)
    return chunks


def process_pdfs_in_directory(directory, chunk_size=1000, chunk_overlap=200):
    """
    处理目录中的所有PDF文件。

    参数:
    directory (str): 目录路径。
    chunk_size (int): 每个文本块的大小。
    chunk_overlap (int): 文本块之间的重叠大小。

    返回:
    tuple: 所有分割后的文本块和对应的索引。
    """
    all_chunks = []
    all_chunks_idx = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_file_path = os.path.join(directory, filename)
            print(f"Processing {pdf_file_path}")

            chunks = new_splitter(pdf_file_path, chunk_size, chunk_overlap)
            all_chunks += chunks
            for i in range(len(chunks)):
                all_chunks_idx.append(f"chunk {i} from {filename}")
    return all_chunks, all_chunks_idx



if __name__ == '__main__':
    pdf_file_dir = './pdf_dir'
    print(process_pdfs_in_directory(pdf_file_dir)[1])

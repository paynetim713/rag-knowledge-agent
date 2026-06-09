"""文档分块: 把长文档切成小块, 适合嵌入和检索

为什么用 RecursiveCharacterTextSplitter:
- 优先按段落切(\\n\\n), 段落超长再按句子(\\n), 还超长按句号
- 比 CharacterTextSplitter 更"懂语义边界", 不容易切断完整意思
- chunk_overlap 让相邻块有 80 字符重叠, 避免关键信息恰好被切在边界
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 中文文档建议加上中文标点
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"分块完成: {len(documents)} 个文档 -> {len(chunks)} 个 chunk")
    return chunks

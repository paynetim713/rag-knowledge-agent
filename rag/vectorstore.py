"""向量存储: 用 ChromaDB 持久化向量, 用 Ollama 嵌入

为什么选 ChromaDB:
- 本地嵌入式, 一个目录就是一个库, 不需要起独立服务
- 自带元数据过滤, 后续要按 source / page 过滤很方便
- API 简洁, LangChain 原生集成

为什么选 nomic-embed-text:
- 体积小(~270MB), 本地能跑
- 中英文都不错
- 768 维, 检索快
"""
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


def _embedder(model_name: str, base_url: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model_name, base_url=base_url)


def build_vectorstore(
    chunks: List[Document],
    persist_directory: str,
    collection_name: str,
    embedding_model: str,
    ollama_base_url: str,
) -> Chroma:
    """从 chunks 构建向量库并持久化. 如果目录已存在, 会被覆盖.

    生产建议: 增量更新而不是全量重建, 但为了演示这里用全量.
    """
    print(f"开始嵌入 {len(chunks)} 个 chunk... (本地推理, 可能需要几分钟)")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_embedder(embedding_model, ollama_base_url),
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    print(f"向量库已持久化到: {persist_directory}")
    return vectorstore


def load_vectorstore(
    persist_directory: str,
    collection_name: str,
    embedding_model: str,
    ollama_base_url: str,
) -> Chroma:
    """从磁盘加载已建好的向量库. 加载时也要传 embedding, 因为查询时要把 query 嵌入."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=_embedder(embedding_model, ollama_base_url),
        persist_directory=persist_directory,
    )


def search(
    vectorstore: Chroma,
    query: str,
    k: int = 3,
    score_threshold: Optional[float] = None,
) -> List[Document]:
    """语义检索: 返回最相关的 k 个 chunk.

    score_threshold: 相似度阈值, 低于阈值的不返回. 防止"完全无关的问题"也被强行匹配.
    """
    if score_threshold is None:
        return vectorstore.similarity_search(query, k=k)
    docs_with_score = vectorstore.similarity_search_with_score(query, k=k)
    return [d for d, score in docs_with_score if score <= score_threshold]

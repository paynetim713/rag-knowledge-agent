"""文档加载: 把 docs/ 目录下的 PDF / TXT 读成 LangChain Document 对象

为什么自己写而不直接用 DirectoryLoader:
1. PDF 我们要保留 page_number, 后面引用时能精确到页
2. TXT 文件名作为 source, 便于引用
3. 出错的文件不能让整个流程崩, 单文件 try/except
"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents(docs_dir: Path) -> List[Document]:
    """递归加载 docs_dir 下所有 .pdf 和 .txt, 返回 Document 列表.

    每个 Document 的 metadata 至少含:
        - source: 文件相对路径
        - page:   PDF 页码(从 0 开始, PyPDFLoader 自带)
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"文档目录不存在: {docs_dir}")

    all_docs: List[Document] = []

    for path in docs_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        rel_source = str(path.relative_to(docs_dir))

        try:
            if suffix == ".pdf":
                # PyPDFLoader 已经按页拆分, 每页一个 Document
                loader = PyPDFLoader(str(path))
                pages = loader.load()
                for d in pages:
                    d.metadata["source"] = rel_source
                all_docs.extend(pages)
                print(f"  [PDF] {rel_source}: {len(pages)} 页")
            elif suffix == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = rel_source
                    d.metadata.setdefault("page", 0)
                all_docs.extend(docs)
                print(f"  [TXT] {rel_source}")
            else:
                continue
        except Exception as e:
            print(f"  [跳过] {rel_source}: {e}")

    print(f"\n共加载 {len(all_docs)} 个文档片段")
    return all_docs

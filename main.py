"""命令行入口

用法:
    python main.py ingest          # 重新索引 docs/ 下所有文档
    python main.py ask "你的问题"   # 单次问答
    python main.py chat            # 进入多轮对话模式
"""
import argparse
import sys
from pathlib import Path

import config
from rag import (
    ask,
    build_qa_chain,
    build_vectorstore,
    load_documents,
    load_vectorstore,
    split_documents,
)


def cmd_ingest():
    """重新索引 docs/ 下所有文档."""
    docs = load_documents(config.DOCS_DIR)
    if not docs:
        print("没有可索引的文档. 把 PDF / TXT 放到 docs/ 目录后重试.")
        return
    chunks = split_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    build_vectorstore(
        chunks=chunks,
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.COLLECTION_NAME,
        embedding_model=config.EMBEDDING_MODEL,
        ollama_base_url=config.OLLAMA_BASE_URL,
    )
    print("\n✅ 索引完成. 现在可以 python main.py ask \"问题\" 或 python main.py chat")


def _load_chain():
    """加载已有向量库 + 构建 QA 链."""
    if not Path(config.CHROMA_PERSIST_DIR).exists():
        print("向量库不存在, 先执行: python main.py ingest")
        sys.exit(1)
    vs = load_vectorstore(
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.COLLECTION_NAME,
        embedding_model=config.EMBEDDING_MODEL,
        ollama_base_url=config.OLLAMA_BASE_URL,
    )
    return build_qa_chain(
        vectorstore=vs,
        llm_model=config.LLM_MODEL,
        ollama_base_url=config.OLLAMA_BASE_URL,
        prompt_template=config.QA_PROMPT_TEMPLATE,
        top_k=config.RETRIEVAL_TOP_K,
        temperature=config.LLM_TEMPERATURE,
    )


def cmd_ask(question: str):
    chain = _load_chain()
    result = ask(chain, question)
    print("\n回答:\n" + result["answer"])
    print("\n引用片段:")
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] {s['source']} (页 {s['page']}): {s['snippet']}")


def cmd_chat():
    """多轮对话: 输入 :q 退出, :clear 清空历史."""
    chain = _load_chain()
    history = []
    print("进入对话模式. 输入 :q 退出, :clear 清空历史.\n")
    while True:
        try:
            q = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q == ":q":
            break
        if q == ":clear":
            history = []
            print("(已清空历史)\n")
            continue
        result = ask(chain, q, history)
        print(f"\nAI> {result['answer']}\n")
        for i, s in enumerate(result["sources"], 1):
            print(f"  📎 [{i}] {s['source']} 页{s['page']}")
        print()
        history.append((q, result["answer"]))


def main():
    parser = argparse.ArgumentParser(description="本地 RAG 知识库问答 Agent")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("ingest", help="重新索引 docs/ 下所有文档")

    p_ask = sub.add_parser("ask", help="单次问答")
    p_ask.add_argument("question", help="你的问题")

    sub.add_parser("chat", help="多轮对话模式")

    args = parser.parse_args()
    if args.cmd == "ingest":
        cmd_ingest()
    elif args.cmd == "ask":
        cmd_ask(args.question)
    elif args.cmd == "chat":
        cmd_chat()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

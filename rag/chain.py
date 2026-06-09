"""RetrievalQA Chain: 把检索 + LLM 拼起来, 支持多轮对话和来源引用

设计思路:
1. 用 RunnablePassthrough 构建 LCEL 链: question -> retrieve -> format prompt -> llm -> answer
2. 同时返回 source_documents, 让前端能展示引用
3. 多轮对话用 history 注入到 prompt, 不用 ConversationBuffer (内存开销可控)
"""
from typing import Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def _format_context(docs: List[Document]) -> str:
    """把检索回来的 chunk 拼成带编号的 context, 让 LLM 能用 [片段N] 引用."""
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "未知")
        page = d.metadata.get("page", "?")
        parts.append(f"[片段{i}] (来源: {src}, 页 {page})\n{d.page_content}")
    return "\n\n".join(parts)


def _format_history(history: List[Tuple[str, str]]) -> str:
    """格式化多轮对话历史. history 是 [(q, a), ...] 列表."""
    if not history:
        return ""
    lines = ["以下是历史对话(供理解上下文, 不要直接复述):"]
    for q, a in history[-3:]:  # 只保留最近 3 轮, 控制 prompt 长度
        lines.append(f"用户: {q}")
        lines.append(f"助手: {a}")
    return "\n".join(lines) + "\n"


def build_qa_chain(
    vectorstore: Chroma,
    llm_model: str,
    ollama_base_url: str,
    prompt_template: str,
    top_k: int = 3,
    temperature: float = 0.1,
):
    """构建一个可调用的 QA 函数. 返回的 callable 签名:
        ask(question, history=[]) -> dict {answer, sources}
    """
    llm = ChatOllama(
        model=llm_model,
        base_url=ollama_base_url,
        temperature=temperature,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 我们自己组装链, 因为要同时拿到 source_documents 和 answer
    prompt = ChatPromptTemplate.from_template(prompt_template)
    parser = StrOutputParser()

    def _invoke(question: str, history: List[Tuple[str, str]] = None) -> Dict:
        history = history or []
        docs = retriever.invoke(question)
        context = _format_context(docs)
        history_text = _format_history(history)

        # 把历史拼到问题里 - 比改 prompt 模板更灵活
        full_question = (history_text + f"用户问题: {question}") if history_text else question

        chain = prompt | llm | parser
        answer = chain.invoke({"context": context, "question": full_question})

        return {
            "answer": answer,
            "sources": [
                {
                    "source": d.metadata.get("source", "未知"),
                    "page": d.metadata.get("page", "?"),
                    "snippet": d.page_content[:200] + ("..." if len(d.page_content) > 200 else ""),
                }
                for d in docs
            ],
        }

    return _invoke


def ask(qa_chain, question: str, history: List[Tuple[str, str]] = None) -> Dict:
    """Thin wrapper, 让上层不直接依赖 qa_chain 的内部结构."""
    return qa_chain(question, history or [])

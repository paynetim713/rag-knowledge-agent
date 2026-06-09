"""Streamlit Web UI - 可视化问答界面

运行: streamlit run app.py
"""
import streamlit as st

import config
from rag import ask, build_qa_chain, load_vectorstore

st.set_page_config(page_title="本地 RAG 知识库", page_icon="📚", layout="wide")


@st.cache_resource
def get_chain():
    """Streamlit 缓存: 避免每次交互都重新加载向量库和模型."""
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


# 侧边栏
with st.sidebar:
    st.title("📚 本地 RAG 知识库")
    st.markdown(
        f"""
**模型配置**
- LLM: `{config.LLM_MODEL}`
- 嵌入: `{config.EMBEDDING_MODEL}`
- Top-K: `{config.RETRIEVAL_TOP_K}`
- 温度: `{config.LLM_TEMPERATURE}`

**特点**
- ✅ 完全离线, 无需 API Key
- ✅ 引用原文 + 页码
- ✅ 多轮对话
- ✅ 防幻觉 Prompt 约束
"""
    )
    if st.button("🧹 清空对话历史"):
        st.session_state.history = []
        st.rerun()


# 主区
st.title("💬 知识库问答")
st.caption("基于 LangChain + Ollama + ChromaDB 的本地 RAG 系统")

if "history" not in st.session_state:
    st.session_state.history = []  # [(q, a, sources), ...]

# 展示历史对话
for q, a, sources in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
        if sources:
            with st.expander(f"📎 引用了 {len(sources)} 个片段", expanded=False):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**[片段{i}]** `{s['source']}` 页 `{s['page']}`")
                    st.text(s["snippet"])

# 输入框
question = st.chat_input("输入你的问题...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索 + 推理..."):
            chain = get_chain()
            # 传入历史(只传 q/a, 不传 sources)
            history_for_chain = [(q, a) for q, a, _ in st.session_state.history]
            result = ask(chain, question, history_for_chain)
        st.markdown(result["answer"])
        with st.expander(f"📎 引用了 {len(result['sources'])} 个片段", expanded=False):
            for i, s in enumerate(result["sources"], 1):
                st.markdown(f"**[片段{i}]** `{s['source']}` 页 `{s['page']}`")
                st.text(s["snippet"])

    st.session_state.history.append((question, result["answer"], result["sources"]))

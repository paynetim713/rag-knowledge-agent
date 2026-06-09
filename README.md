# rag-knowledge-agent

一个本地跑的 RAG 问答工具。把 PDF / TXT 丢进去，用 Ollama 跑的本地模型回答问题，引用原文。

不依赖任何外部 API，文档不出本机。

## 用到的东西

- LangChain — 编排
- Ollama — 本地跑 LLM 和嵌入模型
- ChromaDB — 向量库
- Streamlit — Web UI（可选）

默认模型用的 Qwen3-8B 和 nomic-embed-text，可以在 `config.py` 改。

## 环境

- Python 3.10+
- Ollama（[下载](https://ollama.com/)）
- 内存够跑 8B 模型（推理时大概 6GB）

## 安装

先把 Ollama 装好，然后拉两个模型：

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Python 依赖：

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## 用法

把 PDF / TXT 放到 `docs/` 目录，然后建索引：

```bash
python main.py ingest
```

第一次会比较慢，要把每个 chunk 都过一遍嵌入模型。建完会持久化到 `chroma_db/`，下次直接用不用重建。

问一次：

```bash
python main.py ask "你的问题"
```

多轮对话：

```bash
python main.py chat
```

输入 `:q` 退出，`:clear` 清空历史。

跑 Web UI：

```bash
streamlit run app.py
```

## 目录结构

```
.
├── config.py              配置（模型、chunk 大小、Top-K 等）
├── main.py                CLI
├── app.py                 Streamlit UI
├── rag/
│   ├── loader.py          PDF / TXT 加载
│   ├── splitter.py        分块
│   ├── vectorstore.py     ChromaDB 封装
│   └── chain.py           QA 链
├── docs/                  放文档的地方
└── chroma_db/             向量库（自动生成）
```

## 一些实现细节

**分块**用的 `RecursiveCharacterTextSplitter`，分隔符里加了中文标点（句号、问号、分号等），不然纯英文分隔符切中文文档会有点怪。chunk_size 500，重叠 80。

**检索**默认取 Top-3 拼到 Prompt 里。多了反而稀释模型注意力。

**Prompt** 写死了一条：找不到答案就说找不到，不要瞎编。配合 `temperature=0.1`，幻觉能压不少。

**多轮对话**没用 LangChain 的 `ConversationBufferMemory`，自己存了个 `[(q, a), ...]` 的列表，每次拼最近 3 轮到 Prompt 前面。简单点。

**来源引用**：PyPDFLoader 加载时会自动给每页加 `page` 元数据，检索回来的 chunk 把 source 和 page 拼进 context，让模型用 `[片段N]` 引用。前端能拿到对应的 source 列表。

## 没做的

- Rerank（检索完先 Top-20 再用 cross-encoder 重排 Top-3）
- BM25 关键词检索做混合
- 增量索引（现在加文档要全量重建）
- 流式输出
- OCR（扫描件读不了）

## 配置

`config.py` 里能调的：

| 参数 | 默认 | 说明 |
|---|---|---|
| `CHUNK_SIZE` | 500 | 太小语义不完整，太大检索精度下降 |
| `CHUNK_OVERLAP` | 80 | 防止关键句被切在边界 |
| `RETRIEVAL_TOP_K` | 3 | 检索片段数 |
| `LLM_TEMPERATURE` | 0.1 | RAG 场景越低越好 |
| `EMBEDDING_MODEL` | nomic-embed-text | 768 维，中英文都行 |
| `LLM_MODEL` | qwen3:8b | 也可以换 llama3.1 / mistral 之类 |

## 许可

MIT

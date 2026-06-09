# 本地 RAG 知识库问答 Agent

基于 LangChain + Ollama + ChromaDB 的**全本地** RAG 问答系统。把 PDF / TXT 文档丢进来，用本地大模型语义检索 + 回答，**无需任何 API Key**。

## ✨ 特点

- 🔒 **完全离线**：LLM 推理 + 嵌入 + 向量库全本地，敏感文档不出本机
- 📎 **来源引用**：每个回答标注引用的文档名 + 页码 + 原文片段，可溯源
- 💬 **多轮对话**：保留对话历史，理解上下文
- 🛡️ **防幻觉 Prompt**：严格约束模型只看检索片段，找不到就说"无法回答"
- 🖥️ **两种用法**：命令行（CLI）+ Web UI（Streamlit）

## 🏗️ 架构

```
   ┌────────────┐
   │  PDF/TXT   │
   └─────┬──────┘
         │ PyPDFLoader / TextLoader
         ▼
   ┌────────────────────────┐
   │ RecursiveCharSplitter  │  按段落→句子→词递归切，chunk_size=500
   └─────┬──────────────────┘
         │ List[Document]
         ▼
   ┌────────────────┐      ┌────────────┐
   │ nomic-embed    │─────►│  ChromaDB  │  768 维向量持久化
   └────────────────┘      └─────┬──────┘
                                 │ similarity_search(k=3)
   用户问题 ─────────────────────►│
                                 ▼
   ┌─────────────────────────────────────┐
   │ Prompt:                             │
   │  System: 只看片段回答, 找不到就说没有 │
   │  Context: [片段1] (a.pdf 页3) ...   │
   │  Question: ...                      │
   └────────┬────────────────────────────┘
            │
            ▼
   ┌───────────────┐
   │  Qwen3-8B     │  Ollama 本地推理
   └─────┬─────────┘
         │
         ▼
   {answer, sources: [{source, page, snippet}, ...]}
```

## 📦 环境要求

- Python 3.10+
- Ollama（[下载](https://ollama.com/)）
- 8GB+ 内存（Qwen3-8B 推理需要约 6GB）
- 推荐有 GPU；纯 CPU 也可以跑，推理慢一些

## 🚀 快速开始

### 1. 装 Ollama 并拉模型

```bash
# 装好 Ollama 后启动服务
ollama serve

# 在另一个终端拉模型
ollama pull qwen3:8b           # 推理 LLM（约 5GB）
ollama pull nomic-embed-text   # 嵌入模型（约 270MB）
```

### 2. 装 Python 依赖

```bash
cd rag-knowledge-agent
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. 放文档 + 建索引

把你的 PDF / TXT 文档放到 `docs/` 目录，然后：

```bash
python main.py ingest
```

第一次跑会自动嵌入所有文档（取决于文档数量和 CPU/GPU 性能，几分钟到十几分钟）。完成后 `chroma_db/` 目录就是持久化的向量库。

### 4. 问答

**命令行单次问答：**
```bash
python main.py ask "RAG 系统主要解决哪三个问题?"
```

**命令行多轮对话：**
```bash
python main.py chat
```
输入 `:q` 退出，`:clear` 清空历史。

**Web UI：**
```bash
streamlit run app.py
```
浏览器打开 http://localhost:8501

## 📁 目录结构

```
rag-knowledge-agent/
├── README.md
├── requirements.txt
├── config.py              # 集中配置（模型名、chunk 大小、Top-K 等）
├── main.py                # CLI 入口（ingest / ask / chat）
├── app.py                 # Streamlit Web UI
├── rag/                   # 核心模块
│   ├── __init__.py
│   ├── loader.py          # PDF / TXT 加载，保留 source + page 元数据
│   ├── splitter.py        # RecursiveCharacterTextSplitter，中文友好分隔符
│   ├── vectorstore.py     # ChromaDB 构建 / 加载 / 检索
│   └── chain.py           # RetrievalQA Chain，组装 prompt + LLM + 引用
├── docs/                  # 用户文档（.gitignore 默认不入仓）
└── chroma_db/             # 向量库持久化目录（不入仓）
```

## ⚙️ 配置说明（`config.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `CHUNK_SIZE` | 500 | 每个 chunk 的字符数。太小语义不完整，太大检索精度下降 |
| `CHUNK_OVERLAP` | 80 | 相邻 chunk 重叠字符。防止关键信息被切在边界 |
| `RETRIEVAL_TOP_K` | 3 | 检索最相关的 K 个 chunk 拼进 Prompt |
| `LLM_TEMPERATURE` | 0.1 | 越低越保守。RAG 场景建议 0.0-0.3 减少幻觉 |
| `EMBEDDING_MODEL` | `nomic-embed-text` | 768 维，中英文都不错，体积小 |
| `LLM_MODEL` | `qwen3:8b` | 阿里 Qwen3 8B 量化版，本地能跑 |

## 🧠 设计要点

### 为什么用 RecursiveCharacterTextSplitter
按 `["\n\n", "\n", "。", "！", "？", "；", ...]` 递归切。先按段落，段落太长按句子，句子太长按标点。这样比 `CharacterTextSplitter` 更不容易把完整意思切断。

### 为什么用 ChromaDB
本地嵌入式向量库（不需要起独立服务）。每个 collection 是一个目录，支持元数据过滤、相似度阈值。LangChain 原生集成。生产可以平滑替换成 Milvus / Qdrant / Pinecone。

### 防幻觉的 3 个手段
1. **Prompt 强约束**：明确告诉模型"片段里没有就说没有，不要编"
2. **低温度**：`temperature=0.1`
3. **来源引用**：每个回答都带 [片段N] 引用，让用户能核对

### 多轮对话怎么做
不用 `ConversationBufferMemory`（容易爆 Prompt）。手动维护 `[(q, a), ...]` 列表，每次问答把**最近 3 轮**拼到 Prompt 前面。简单、可控。

## 🧪 示例

```bash
$ python main.py ingest
  [TXT] sample.txt
共加载 1 个文档片段
分块完成: 1 个文档 -> 3 个 chunk
开始嵌入 3 个 chunk...
向量库已持久化到: ./chroma_db
✅ 索引完成.

$ python main.py ask "RAG 主要解决哪三个问题?"

回答:
根据参考资料, RAG 主要解决三个问题:
1. 大模型训练数据有截止日期, 无法获取最新信息 [片段1]
2. 大模型对私有数据一无所知 [片段1]
3. 大模型回答容易产生幻觉, RAG 把原始材料喂进去, 模型有据可依 [片段1]

引用片段:
  [1] sample.txt (页 0): RAG 主要解决三个问题: 1. 大模型训练数据有截止日期...
```

## 🔮 未来优化方向

- [ ] **Rerank**：检索 Top-20 后用 Cross-Encoder 重排 Top-3，准确率提升明显
- [ ] **Hybrid Search**：语义检索 + BM25 关键词检索融合，提升召回
- [ ] **增量更新**：新增文档时只嵌入新文档，不全量重建
- [ ] **流式输出**：LLM 边生成边显示，提升体验
- [ ] **多模态**：支持图片（截图、扫描件 + OCR）
- [ ] **Query Rewrite**：对模糊问题用 LLM 改写后再检索

## 📜 License

MIT

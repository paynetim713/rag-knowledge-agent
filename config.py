"""集中配置 - 所有可调参数在这里"""
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent

# 文档目录(放 PDF / TXT)
DOCS_DIR = BASE_DIR / "docs"

# 向量数据库持久化目录
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")

# Chroma 集合名称
COLLECTION_NAME = "knowledge_base"

# Ollama 服务地址
OLLAMA_BASE_URL = "http://localhost:11434"

# 嵌入模型(nomic-embed-text 1.5: 768 维, 中英文都不错, 体积 ~270MB)
EMBEDDING_MODEL = "nomic-embed-text"

# 推理 LLM(简历写的 Qwen3-8B)
LLM_MODEL = "qwen3:8b"

# 文档分块: 按字符递归切, 优先按段落 -> 句子 -> 单词
CHUNK_SIZE = 500          # 每块字符数
CHUNK_OVERLAP = 80        # 相邻块重叠, 防止切断关键信息

# 检索 Top-K: 每次取最相关的 K 个片段拼进 Prompt
RETRIEVAL_TOP_K = 3

# LLM 生成参数
LLM_TEMPERATURE = 0.1     # 越低越保守, 减少幻觉
LLM_MAX_TOKENS = 1024

# Prompt 模板 - 强约束让模型只看上下文回答
QA_PROMPT_TEMPLATE = """你是一个严谨的知识库问答助手。请只根据下面提供的「参考片段」回答用户问题。

规则:
1. 如果参考片段里没有答案, 直接回答"根据现有资料无法回答该问题"——不要编造。
2. 回答要简洁、准确, 用中文输出。
3. 如果引用了片段内容, 在句末用 [片段N] 标注来源。

参考片段:
{context}

用户问题: {question}

回答:"""

# 📚 Enterprise RAG System (企业级大模型知识库问答系统)

基于 LangChain、ChromaDB 和大语言模型（LLM）构建的本地化企业知识库问答系统。本项目实现了从文档解析、向量化存储到带有多轮对话记忆的检索问答（RAG）全流程。

## 🌟 核心特性 (Features)
- **多格式文档支持**：无缝解析 `PDF`、`Word (.docx)` 和 `TXT` 文本文件。
- **高级切片策略**：采用 `RecursiveCharacterTextSplitter` 进行按语义边界重叠切片，防止上下文截断，优化检索精度。
- **本地化 Embedding**：内置 HuggingFace `sentence-transformers` 轻量级多语言模型，文档向量化全部在本地完成，保护企业数据隐私。
- **持久化向量数据库**：集成 `ChromaDB`，一次构建，持续可用。
- **智能对话记忆**：采用 LangChain LCEL 表达式重构检索链，实现历史上下文感知的独立问题重写（History-Aware Retriever），支持丝滑的多轮问答。
- **交互式 Web UI**：基于 Streamlit 构建现代化的响应式 Web 界面。

## 🛠️ 技术栈 (Tech Stack)
- **Framework**: [LangChain](https://python.langchain.com/) (LCEL Architecture)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **LLM**: 支持 OpenAI GPT 系列，或 DeepSeek, Qwen 等兼容 OpenAI API 格式的模型
- **Frontend**: [Streamlit](https://streamlit.io/)

## 🚀 快速开始 (Quick Start)

### 环境要求
- Python 3.9+
- 可选：CUDA 环境（加速本地 Embedding 模型推理）

### 1. 克隆仓库
```bash
https://github.com/kinglintion/Enterprise-RAG-System.git
cd Enterprise-RAG-System
```

### 2. 创建虚拟环境（推荐）
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
# source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

> **注意**：`requirements.txt` 中的 `sentence-transformers` 和 `chromadb` 会自动下载依赖的 PyTorch 和 ONNX 运行时。如果网络较慢，建议先配置国内 PyPI 镜像源（如 `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`）。

### 4. 启动系统
```bash
streamlit run app.py
```

### 5. 使用流程
1. 浏览器打开 Streamlit 提供的本地地址（默认 `http://localhost:8501`）
2. 在侧边栏填写 **API Key** 和 **Model Name**（支持 OpenAI / DeepSeek / 通义千问等兼容 API）
3. 上传 PDF、Word 或 TXT 文档，点击"构建/更新知识库"
4. 在聊天框提问，系统会基于文档内容进行多轮对话

## 📁 项目结构
```
RAG-system/
├── app.py              # Streamlit Web 应用入口
├── rag_core.py         # RAG 核心逻辑（文档加载、切片、向量化、检索链）
├── requirements.txt    # Python 依赖清单
├── readme.md           # 项目说明文档
├── data/               # 上传的文档存放目录（自动创建）
├── chroma_db/          # Chroma 向量数据库持久化目录（自动生成）
└── __pycache__/        # Python 缓存目录
```

## 🧠 系统架构 (Architecture)
```
用户提问
    │
    ▼
┌─────────────────────────────────────────────────┐
│  1. History-Aware Rewriter                      │
│     结合对话历史，将模糊问题重写为独立问题         │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  2. 语义检索 (Retriever)                        │
│     ChromaDB 中 Top-8 最相关文档片段             │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  3. Stuff Documents Chain                       │
│     将检索片段组装为 Prompt Context              │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  4. LLM 生成回答                                │
│     基于 Context + 历史 + 当前问题               │
└─────────────────────────────────────────────────┘
    │
    ▼
   返回答案
```

## ⚙️ 详细配置说明

### 支持的 LLM 后端
| 服务商 | Base URL | 推荐 Model |
|--------|----------|------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-3.5-turbo`, `gpt-4` |
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
| 通义千问 (阿里云) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus`, `qwen-max` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4` |

### 文档处理参数
- **切片大小 (chunk_size)**: 1500 字符
- **切片重叠 (chunk_overlap)**: 300 字符（防止核心概念被切断）
- **检索数量 (retriever k)**: 8 个片段
- **Embedding 模型**: `paraphrase-multilingual-MiniLM-L12-v2`（多语言，支持中英文混排）


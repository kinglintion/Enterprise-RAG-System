import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class RAGSystem:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None

    def process_and_store_document(self, file_path):
        """处理多格式文档，切片并存入 Chroma 向量数据库"""
        ext = os.path.splitext(file_path)[1].lower()
        
        # 1. 多格式文档加载
        if ext == '.pdf':
            loader = PyMuPDFLoader(file_path) 
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,      # 每个切片约1500字符 (应对较长段落)
            chunk_overlap=300,    # 重叠300字符，防止核心概念（如创新点）被切断
            separators=["\n\n", "\n", ".", "。", " ", ""] 
        )
        chunks = text_splitter.split_documents(documents)

        # 3. 向量化并存储到 Chroma DB
        self.vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            persist_directory=self.persist_directory
        )
        
        # 初始化检索器
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 8})
        return f"成功处理并向量化 {len(chunks)} 个文本块！"

    def init_qa_chain(self, api_key, base_url="https://api.openai.com/v1", model_name="gpt-3.5-turbo"):
        """初始化带有历史记忆的检索问答链"""
        if not self.retriever:
            # 如果程序重启，尝试加载已有的本地数据库
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 8})
            else:
                raise ValueError("请先上传文档构建知识库！")

        # 初始化大模型 
        llm = ChatOpenAI(
            api_key=api_key, 
            base_url=base_url, 
            model=model_name, 
            temperature=0.3
        )

        #对话记忆处理 
        # 1. 历史问题改写 Prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "给定对话历史记录和用户最新提出的问题。请将最新问题重写为一个独立的问题，以便在不参考历史记录的情况下也能理解。"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, self.retriever, contextualize_q_prompt)

        # 2. 最终问答 Prompt 
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的学术和企业知识库AI研究员。
            请基于以下检索到的文档片段（Context）回答用户问题。
            
            阅读策略：
            1. 用户可能会用中文提问，但文档可能是英文。请注意将中文术语（如“创新点”、“方法”、“结论”）映射到英文学术词汇（如 Contributions, Novelty, Methodology, Conclusion）。
            2. 仔细阅读提供的片段，提取出核心逻辑。如果片段比较零散，请尝试整合它们。
            3. 回答要结构化、清晰易懂。
            4. 绝对忠于原文内容。如果提供的片段中完全没有答案，请明确回答“根据现有检索到的文档片段，未能找到该信息”，绝不能自己编造。
            
            背景内容 (Context)：
            {context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # 3. 组合 RAG 链
        self.qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def ask(self, question, chat_history):
        """执行问答并返回结果"""
        if not self.qa_chain:
            raise ValueError("QA Chain 未初始化，请检查 API Key 和 模型配置。")
        
        response = self.qa_chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        return response["answer"]
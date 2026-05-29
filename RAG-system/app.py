import streamlit as st
import os
from rag_core import RAGSystem
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="企业智能知识库助手", page_icon="📚", layout="wide")

# 初始化后端系统
@st.cache_resource
def get_rag_system():
    return RAGSystem()

rag_system = get_rag_system()

# 初始化 Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False

# 侧边栏：配置与文件上传 
with st.sidebar:
    st.header("⚙️ 系统配置")
    api_key = st.text_input("大模型 API Key", type="password")
    # 为了方便国内用户，这里预留了 Base URL 选项，可以用通义千问/DeepSeek等
    base_url = st.text_input("API Base URL (可选)", value="https://api.openai.com/v1")
    model_name = st.text_input("Model Name", value="gpt-3.5-turbo")
    
    st.markdown("---")
    st.header("📂 知识库构建")
    uploaded_file = st.file_uploader("上传企业文档 (PDF/TXT/Word)", type=['pdf', 'txt', 'docx'])
    
    if st.button("构建/更新知识库"):
        if not uploaded_file:
            st.warning("请先上传文件！")
        elif not api_key:
            st.warning("请填写 API Key！")
        else:
            with st.spinner("正在处理文档 (切片、Embedding、存入 Chroma)..."):
                # 临时保存文件
                os.makedirs("data", exist_ok=True)
                temp_path = os.path.join("data", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # 处理文档
                result = rag_system.process_and_store_document(temp_path)
                
                # 初始化 QA Chain
                rag_system.init_qa_chain(api_key, base_url, model_name)
                st.session_state.system_initialized = True
                
                st.success(result)
                st.success("知识库构建完成，开始提问吧！")

# 主界面：聊天交互 
st.title("📚 企业知识库智能问答系统 (RAG)")
st.markdown("上传文档后，即可基于文档内容与 AI 进行多轮对话。")

# 显示历史记录
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# 聊天输入框
if prompt := st.chat_input("请输入您的问题..."):
    if not st.session_state.system_initialized:
        st.error("请先在侧边栏上传文档并构建知识库！")
    else:
        # 显示用户输入
        st.chat_message("user").write(prompt)
        
        # 调用大模型生成回答
        with st.spinner("检索文档并思考中..."):
            try:
                answer = rag_system.ask(prompt, st.session_state.chat_history)
                st.chat_message("assistant").write(answer)
                
                # 更新记忆
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=answer))
            except Exception as e:
                st.error(f"发生错误: {str(e)}")
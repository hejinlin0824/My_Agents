import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chat_models import init_chat_model
import os
import shutil
from dotenv import load_dotenv

load_dotenv(override=True)

# --- 配置 ---
DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
dashscope_api_key = os.getenv("dashscope_api_key")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
FAISS_DB_PATH = "faiss_db"

# --- 初始化模型和嵌入 ---
# 使用 st.cache_resource 来缓存这些昂贵的对象
@st.cache_resource
def get_embeddings_model():
    return DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=dashscope_api_key)

@st.cache_resource
def get_llm():
    return init_chat_model("deepseek-chat", model_provider="deepseek")

embeddings = get_embeddings_model()
llm = get_llm()

# --- 核心功能函数 ---
def pdf_read(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def vector_store(text_chunks):
    if not text_chunks:
        st.error("❌ 文本分块为空，无法创建向量数据库。")
        return
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_DB_PATH)
    except Exception as e:
        st.error(f"❌ 创建向量数据库失败: {e}")

def get_qa_agent(retriever_tool):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是AI助手。请根据提供的上下文(context)来严谨地回答问题。
        - 确保提供所有相关的细节。
        - 如果答案在上下文中无法找到，请明确说明：“根据提供的文档，我无法找到相关答案。”
        - 不要编造信息。"""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    tools = [retriever_tool]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def check_database_exists():
    return os.path.exists(FAISS_DB_PATH) and os.path.exists(os.path.join(FAISS_DB_PATH, "index.faiss"))

# --- 主应用逻辑 ---
def main():
    st.set_page_config(page_title="🤖 WhatsYourName", page_icon="🤖")
    st.header("🤖 WhatsYourName \n To Simplify Your Life")

    # --- 侧边栏 ---
    with st.sidebar:
        st.title("📁 文档管理")
        
        if check_database_exists():
            st.success("✅ 数据库已就绪，可以开始提问。")
        else:
            st.info("📝 请先上传PDF并处理。")
        
        st.markdown("---")
        
        pdf_docs = st.file_uploader(
            "📎 上传你的PDF文件", 
            accept_multiple_files=True,
            type=['pdf'],
            help="你可以上传一个或多个PDF文件。"
        )
        
        if st.button("🚀 处理文档", use_container_width=True, disabled=not pdf_docs):
            with st.spinner("⏳ 正在读取和处理PDF..."):
                try:
                    raw_text = pdf_read(pdf_docs)
                    if not raw_text.strip():
                        st.error("❌ 无法从PDF中提取任何文本。请检查PDF文件是否为可选中复制的文本格式，而非纯图片。")
                        return

                    text_chunks = get_chunks(raw_text)
                    st.info(f"✅ 文本分割完成，共 {len(text_chunks)} 个片段。")
                    
                    vector_store(text_chunks)
                    st.success("✅ PDF处理完成！向量数据库已创建。")
                    st.balloons()
                    
                    # 清空之前的聊天记录
                    if "messages" in st.session_state:
                        st.session_state.messages = []
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 处理过程中出现错误: {e}")
        
        if st.button("🗑️ 清除数据库", use_container_width=True):
            if os.path.exists(FAISS_DB_PATH):
                shutil.rmtree(FAISS_DB_PATH)
                st.success("数据库已成功清除！")
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()
            else:
                st.warning("数据库不存在。")
        
        with st.expander("💡 使用说明"):
            st.markdown("""
            1.  **上传**: 在侧边栏上传一个或多个PDF文件。
            2.  **处理**: 点击“处理文档”按钮，应用会将文档内容存入知识库。
            3.  **提问**: 在主聊天窗口输入问题，AI将根据文档内容回答。
            4.  **清除**: 如果想更换文档，可以先“清除数据库”。
            """)

    # --- 聊天界面 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not check_database_exists():
        st.info("请先在左侧边栏上传并处理您的PDF文档，然后才能开始提问。")
    else:
        if user_question := st.chat_input("请在此输入你关于文档的问题..."):
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("🤔 AI正在思考..."):
                    try:
                        # 加载FAISS数据库并创建retriever工具
                        db = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
                        retriever = db.as_retriever()
                        retriever_tool = create_retriever_tool(retriever, "pdf_content_retriever", "在PDF文档中搜索和检索相关信息以回答用户问题。")

                        # 创建Agent
                        qa_agent = get_qa_agent(retriever_tool)
                        
                        # 准备聊天记录
                        chat_history = []
                        for msg in st.session_state.messages[:-1]: # 获取除最后一个问题外的所有历史记录
                            if msg["role"] == "user":
                                chat_history.append(HumanMessage(content=msg["content"]))
                            elif msg["role"] == "assistant":
                                chat_history.append(AIMessage(content=msg["content"]))
                        
                        # 调用Agent获取回答
                        response = qa_agent.invoke({
                            "input": user_question,
                            "chat_history": chat_history
                        })
                        
                        answer = response.get('output', '抱歉，我无法生成回答。')
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                    except Exception as e:
                        error_message = f"❌ 回答生成失败: {e}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()

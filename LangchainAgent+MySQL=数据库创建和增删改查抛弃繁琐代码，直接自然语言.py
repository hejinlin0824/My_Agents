import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_openai import ChatOpenAI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载 .env 文件中的环境变量
load_dotenv()

def get_db_connection_string(db_name=None):
    """
    根据是否提供数据库名称，构建数据库连接字符串

    参数:
    db_name (str, optional): 数据库名称。默认为None。

    返回:
    str: 数据库连接字符串
    """
    # 从环境变量中获取数据库连接信息
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")

    # 检查数据库连接信息是否完整
    if not all([db_user, db_password, db_host, db_port]):
        raise ValueError("数据库连接信息不完整，请检查 .env 文件。")
    
    # 根据是否提供了数据库名称，构建并返回相应的数据库连接字符串
    if db_name:
        return f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    else:
        return f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}"
def get_available_databases():
    """获取所有可用的数据库名称"""
    try:
        engine = create_engine(get_db_connection_string())
        inspector = inspect(engine)
        db_names = inspector.get_schema_names()
        # 过滤掉系统数据库
        system_dbs = ['information_schema', 'mysql', 'performance_schema', 'sys']
        available_dbs = [db for db in db_names if db not in system_dbs]
        logging.info(f"发现可用数据库: {available_dbs}")
        return available_dbs
    except Exception as e:
        logging.error(f"连接数据库失败，请检查连接信息和数据库服务是否正常: {e}")
        return []

def main():
    """主函数"""
    # 设置 DeepSeek API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logging.error("未找到 DEEPSEEK_API_KEY，请检查 .env 文件。")
        return

    # 1. 选择数据库
    databases = get_available_databases()
    if not databases:
        return

    print("检测到以下数据库，请选择一个进行操作：")
    for i, db_name in enumerate(databases):
        print(f"{i + 1}. {db_name}")

    try:
        choice = int(input("🤖：请输入数据库编号: ")) - 1
        if not 0 <= choice < len(databases):
            print("🤖：无效的选择。")
            return
        selected_db = databases[choice]
        logging.info(f"用户选择了数据库: {selected_db}")
    except ValueError:
        print("请输入有效的数字。")
        return

    # 2. 初始化 Langchain SQL Agent
    try:
        db_uri = get_db_connection_string(selected_db)
        db = SQLDatabase.from_uri(db_uri)
        
        # 需要将 DeepSeek 的 base_url 设置为 os.environ["OPENAI_API_BASE"]
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

        llm = ChatOpenAI(model="deepseek-chat", temperature=0)
        
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="openai-tools",
            prefix=f"你是一个MySQL专家。给定一个输入问题，创建一个语法正确的MySQL查询来运行，然后查看查询结果并返回答案。除非用户明确要求查询特定数量的结果，否则查询所有结果。你可以对数据库中的表进行排序以查看可以查询哪些表。在查询列之前，你必须先查询表中的所有列。\n\n当前操作的数据库是 `{selected_db}`.除非必须用英文，否则所有回答用中文回复。"
        )
        logging.info("SQL Agent 初始化成功。")
    except Exception as e:
        logging.error(f"初始化 SQL Agent 失败: {e}")
        return

    # 3. 与 Agent 交互
    print("\nSQL Agent 已准备就绪，可以开始提问了（输入 '退出' 来结束）。")
    while True:
        user_input = input("🤖：你想做什么? > ")
        if user_input.lower() == '退出':
            print("🤖：再见！")
            break
        
        if not user_input.strip():
            continue

        try:
            logging.info(f"用户输入: {user_input}")
            result = agent_executor.invoke(user_input)
            print("\n🤖[Agent输出]:")
            print(result.get('output', '没有获取到结果。'))
            print("-" * 20)
        except Exception as e:
            logging.error(f"Agent 执行出错: {e}")
            print("抱歉，在处理您的请求时遇到了问题，请检查日志获取详细信息。")

if __name__ == "__main__":
    main()

# My_Agents  
> 我的一些 agents，提高生产力的同时还能学习。  
> 名字即功能，直接看名字就知道它能干什么。

---

## 1. 创建并激活虚拟环境  
```bash
python -m venv llm
```

## 2. 激活虚拟环境
Windows:
```bash
.\llm\Scripts\activate
```

Linux/MacOS
```bash
source llm/bin/activate
```

## 3. 安装依赖库
```bash
pip install -r requirements.txt
```
## 4. 运行 Agent
直接执行目标脚本即可，例如：
```bash
python your_agent_script.py
```

# 特殊说明：RAG Agent
RAG 相关脚本需通过 Streamlit 运行：
```bash
streamlit run ./rag_agent.py
```

## 项目说明
每个 Agent 设计为独立功能模块，名称即核心用途。
建议在虚拟环境中运行以避免依赖冲突。



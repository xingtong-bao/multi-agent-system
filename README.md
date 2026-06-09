# 🤖 Multi-Agent 协作研究系统

基于 **LangGraph StateGraph + DeepSeek V3** 构建的三Agent协作系统。

## Agent架构
```
用户任务 → 🧠规划Agent → 🔍执行Agent → 📝审查Agent → 最终报告
```

## 技术栈
- **图编排**：LangGraph StateGraph
- **LLM**：DeepSeek V3
- **搜索**：DuckDuckGo
- **界面**：Gradio

## 快速开始
```bash
pip3 install -r requirements.txt
cp env.example .env  # 填入 DeepSeek API Key
python3 app.py
```
访问 http://127.0.0.1:7862

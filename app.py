import os
import json
import gradio as gr
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from openai import OpenAI
from duckduckgo_search import DDGS

load_dotenv()

DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-chat"


def get_client():
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )


def call_deepseek(prompt: str, system: str = "") -> str:
    """Direct DeepSeek V3 call."""
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=2000
    )
    return response.choices[0].message.content


def web_search(query: str) -> str:
    """DuckDuckGo search tool."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"未找到关于「{query}」的搜索结果"
        output = ""
        for i, r in enumerate(results, 1):
            output += f"[{i}] {r.get('title','')}\n{r.get('body','')}\n\n"
        return output.strip()
    except Exception as e:
        return f"搜索出错：{str(e)}"


# ── LangGraph State ───────────────────────────────────────────────
class AgentState(TypedDict):
    task:           str
    subtasks:       List[str]
    search_results: List[str]
    final_report:   str
    log:            List[str]   # progress log for UI


# ── Node 1: Planner Agent ────────────────────────────────────────
def planner_node(state: AgentState) -> AgentState:
    """
    规划Agent：把用户任务拆解为3个具体子任务
    """
    task = state["task"]
    log  = state.get("log", [])
    log.append("🧠  [规划Agent] 正在分析任务，拆解子任务...")

    system = "你是一个任务规划专家。将用户的研究任务拆解为3个具体、可搜索的子任务。只返回JSON格式，不要其他内容。"
    prompt = f"""
请将以下研究任务拆解为3个子任务：
任务：{task}

返回格式（严格JSON）：
{{"subtasks": ["子任务1", "子任务2", "子任务3"]}}
"""
    response = call_deepseek(prompt, system)

    try:
        # Clean JSON from response
        clean = response.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data     = json.loads(clean.strip())
        subtasks = data.get("subtasks", [])
    except Exception:
        # Fallback: generate subtasks manually
        subtasks = [
            f"{task} - 基本概念与背景",
            f"{task} - 最新发展动态",
            f"{task} - 未来趋势与影响"
        ]

    for i, s in enumerate(subtasks, 1):
        log.append(f"   📋 子任务{i}：{s}")

    return {**state, "subtasks": subtasks, "log": log}


# ── Node 2: Executor Agent ────────────────────────────────────────
def executor_node(state: AgentState) -> AgentState:
    """
    执行Agent：对每个子任务进行网络搜索，收集信息
    """
    subtasks = state["subtasks"]
    log      = state.get("log", [])
    results  = []

    log.append("🔍  [执行Agent] 开始对每个子任务进行网络搜索...")

    for i, subtask in enumerate(subtasks, 1):
        log.append(f"   🔎 搜索子任务{i}：{subtask}")
        result = web_search(subtask)
        results.append(f"### 子任务{i}：{subtask}\n\n{result}")

    log.append(f"   ✅ 搜索完成，共收集 {len(results)} 条资料")

    return {**state, "search_results": results, "log": log}


# ── Node 3: Reviewer Agent ────────────────────────────────────────
def reviewer_node(state: AgentState) -> AgentState:
    """
    审查Agent：汇总所有搜索结果，生成最终研究报告
    """
    task           = state["task"]
    subtasks       = state["subtasks"]
    search_results = state["search_results"]
    log            = state.get("log", [])

    log.append("📝  [审查Agent] 正在汇总资料，生成最终报告...")

    all_results = "\n\n---\n\n".join(search_results)
    subtask_list = "\n".join([f"{i+1}. {s}" for i, s in enumerate(subtasks)])

    prompt = f"""你是一个专业研究分析师。请基于以下资料，为课题「{task}」生成一份完整的研究报告。

## 研究子任务：
{subtask_list}

## 收集到的资料：
{all_results}

请生成结构化报告：

# 研究报告：{task}

## 一、概述
（2-3句话概括）

## 二、核心发现
（针对每个子任务的关键发现，用 • 列出）

## 三、深度分析
（综合所有资料的深入分析）

## 四、结论与建议
（总结 + 可操作的建议）

---
*本报告由 LangGraph Multi-Agent System + DeepSeek V3 自动生成*
*参与Agent：规划Agent → 执行Agent → 审查Agent*"""

    report = call_deepseek(prompt)
    log.append("   ✅ 报告生成完成！")

    return {**state, "final_report": report, "log": log}


# ── Build LangGraph ───────────────────────────────────────────────
def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner",  planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reviewer", reviewer_node)

    # Define flow: planner → executor → reviewer → END
    workflow.set_entry_point("planner")
    workflow.add_edge("planner",  "executor")
    workflow.add_edge("executor", "reviewer")
    workflow.add_edge("reviewer", END)

    return workflow.compile()


# ── Research runner ───────────────────────────────────────────────
def run_multi_agent(task: str, history: list):
    if not task.strip():
        return history, ""

    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_key_here":
        history.append((task, "❌  请先在 .env 文件中配置 DEEPSEEK_API_KEY"))
        yield history, ""
        return

    history.append((task, "🚀  LangGraph Multi-Agent 系统启动中..."))
    yield history, ""

    try:
        graph = build_graph()

        # Run the graph
        initial_state: AgentState = {
            "task":           task,
            "subtasks":       [],
            "search_results": [],
            "final_report":   "",
            "log":            []
        }

        # Stream progress
        final_state = None
        for step in graph.stream(initial_state):
            node_name  = list(step.keys())[0]
            node_state = step[node_name]
            log        = node_state.get("log", [])

            node_emoji = {
                "planner":  "🧠 规划Agent",
                "executor": "🔍 执行Agent",
                "reviewer": "📝 审查Agent"
            }

            progress = f"**▶ {node_emoji.get(node_name, node_name)} 运行中...**\n\n"
            progress += "\n".join(log)
            history[-1] = (task, progress)
            yield history, ""

            final_state = node_state

        # Build final output
        if final_state:
            log    = final_state.get("log", [])
            report = final_state.get("final_report", "")

            process_log = "### 🤖 Multi-Agent 执行过程\n\n"
            process_log += "\n".join(log)
            process_log += "\n\n---\n\n"

            history[-1] = (task, process_log + report)
            yield history, ""

    except Exception as e:
        history[-1] = (task, f"❌  出错：{str(e)}")
        yield history, ""


# ── Gradio UI ─────────────────────────────────────────────────────
with gr.Blocks(
    title="Multi-Agent 协作系统",
    theme=gr.themes.Soft(),
    css="footer { display: none !important; }"
) as demo:

    gr.Markdown("""
    # 🤖 Multi-Agent 协作研究系统
    **LangGraph StateGraph + DeepSeek V3 · 三Agent协作架构**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            ### ⚙️ Agent 架构
            ```
            用户任务
               ↓
            🧠 规划Agent
            （任务拆解）
               ↓
            🔍 执行Agent
            （网络搜索）
               ↓
            📝 审查Agent
            （报告生成）
               ↓
            最终报告
            ```

            ### 💡 任务示例
            - 2026年中国AI独角兽投资机会
            - MLOps工程师的核心技能要求
            - DeepSeek vs OpenAI技术对比
            - LangGraph在生产环境的应用

            ### ⏱️ 预计时间
            每次约 40～60 秒
            """)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Multi-Agent 研究报告",
                height=520,
                bubble_full_width=False
            )
            with gr.Row():
                task_input = gr.Textbox(
                    label="输入研究任务",
                    placeholder="例如：分析2026年中国AI行业的投资机会",
                    lines=2,
                    scale=4
                )
                with gr.Column(scale=1):
                    submit_btn = gr.Button("启动Agent 🚀", variant="primary")
                    clear_btn  = gr.Button("清空 🗑️")

    submit_btn.click(run_multi_agent, [task_input, chatbot], [chatbot, task_input])
    task_input.submit(run_multi_agent, [task_input, chatbot], [chatbot, task_input])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, task_input])


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        show_error=True
    )

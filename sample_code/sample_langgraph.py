
#실습용 AOAI 환경변수 읽기
import os

AOAI_ENDPOINT=os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY=os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O=os.getenv("AOAI_DEPLOY_GPT4O")
AOAI_DEPLOY_GPT4O_MINI=os.getenv("AOAI_DEPLOY_GPT4O_MINI")
AOAI_DEPLOY_EMBED_3_LARGE=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE")
AOAI_DEPLOY_EMBED_3_SMALL=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL")
AOAI_DEPLOY_EMBED_ADA=os.getenv("AOAI_DEPLOY_EMBED_ADA")

print("AOAI_ENDPOINT:", AOAI_ENDPOINT)
print("AOAI_API_KEY:", AOAI_API_KEY)
print("AOAI_DEPLOY_GPT4O:", AOAI_DEPLOY_GPT4O)
print("AOAI_DEPLOY_GPT4O_MINI:", AOAI_DEPLOY_GPT4O_MINI)
print("AOAI_DEPLOY_EMBED_3_LARGE:", AOAI_DEPLOY_EMBED_3_LARGE)
print("AOAI_DEPLOY_EMBED_3_SMALL:", AOAI_DEPLOY_EMBED_3_SMALL)
print("AOAI_DEPLOY_EMBED_ADA:", AOAI_DEPLOY_EMBED_ADA)

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_DEPLOY_GPT4O_MINI,
    api_version="2024-10-21",
    api_key=AOAI_API_KEY
)

from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

@tool
def get_cafeteria_menu(day: str|None):
    """
    주어진 요일의 구내식당 메뉴를 조회하는 함수.
    day 가 None 일 경우 이번주 메뉴 조회

    :param day: 조회할 요일 (예: '월요일', '화요일' 등)
    :return: JSON 응답을 파이썬 딕셔너리로 변환하여 반환
    """
    url = "http://52.141.29.94:8800/cafeteria-menu"
    params = {"day": day}
    headers = {"accept": "application/json"}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        return response.json()  # JSON 응답을 파이썬 딕셔너리로 변환
    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 오류 발생: {e}")
        return None
        
@tool
def get_schedule():
    """스케쥴 조회"""
    url = "http://52.141.29.94:8800/schedule"
    headers = {"accept": "application/json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        return response.json() 
    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 오류 발생: {e}")
        return None


members = ["cafeteria", "schedule"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

# Literally Router. Designates which node should be next with fixed options
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options]

class State(MessagesState):
    next: str

# 입력: state — State 타입(내부에 메시지 리스트 있음)
# 출력: Command 객체, 제네릭에 Literal로 members + "__end__" 가능한 값 제한
def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    # 시스템 메시지(역할 설명 등)를 맨 앞에 두고,
	# 기존 대화 메시지 리스트(state["messages"])와 합쳐서 LLM에 보낼 messages 리스트 생성
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    # Router 타입으로 구조화된 출력(즉, next 필드에 "cafeteria", "schedule", "FINISH" 중 하나 포함)을 기대
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})
 

# 에이전트를 자동으로 만들어줌. 모델과, 툴과, 할 일을 정해줌
# 더 전문적인 분석을 위해 o4-mini를 붙여도 되고, 빠른 응답을 위해 4.1-mini를 붙일 수도 있다!
cafeteria_agent = create_react_agent(
    llm, tools=[get_cafeteria_menu], prompt="당신은 구내식당을 관리하는 영양사입니다. 사용자에게 이번 주의 식단을 알려줄 수 있습니다."
)

def cafeteria_node(state: State) -> Command[Literal["supervisor"]]:
    result = cafeteria_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="cafeteria")
            ]
        },
        goto="supervisor",
    )

schedule_agent = create_react_agent(
    llm, tools=[get_schedule], prompt="당신은 사용자의 일정을 관리하는 비서입니다. 사용자에게 현재 남아있는 일정을 안내합니다."
)

def schedule_node(state: State) -> Command[Literal["supervisor"]]:
    result = schedule_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="schedule")
            ]
        },
        goto="supervisor",
    )

def main():
    print("Welcome to the LangGraph Chatbot!")
    print("Type 'exit' to quit.\n")

    # Initialize state
    state = {"messages": [], "next": "supervisor"}

    # Build the graph
    workflow = StateGraph(State)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("cafeteria", cafeteria_node)
    workflow.add_node("schedule", schedule_node)
    workflow.add_edge("supervisor", "cafeteria")
    workflow.add_edge("supervisor", "schedule")
    workflow.add_edge("supervisor", END)
    workflow.add_edge("cafeteria", "supervisor")
    workflow.add_edge("schedule", "supervisor")
    workflow.set_entry_point("supervisor")
    app = workflow.compile()

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break

        state["messages"].append({"role": "user", "content": user_input})
        for step in app.stream(state):
            state = step["state"]
            # Print only new assistant messages
            for msg in state["messages"]:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    print(f"Bot: {msg['content']}")
                elif hasattr(msg, "name") and msg.name in members:
                    print(f"{msg.name.capitalize()}: {msg.content}")

        # Remove all but the last message to keep context short
        state["messages"] = state["messages"][-5:]

if __name__ == "__main__":
    main()
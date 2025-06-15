# LangGraph 기반 멀티 에이전트 구현

## 프로젝트 개요
이 프로젝트는 LangGraph를 활용하여 멀티 에이전트를 설계하고 구현하는 예제입니다. 멀티 에이전트는 서로 협력하여 다양한 사용자의 요청을 처리하며, 이를 통해 복잡한 AI 워크플로우를 효율적으로 운영할 수 있습니다. 

주요 목표:
- LangChain과 LangGraph를 활용하여 에이전트의 병렬 처리 및 협업 기능 구현
- 멀티 에이전트 시스템에 필요한 상태 관리와 흐름 제어 설계
- 사용자가 입력한 질문에 대한 최적의 응답을 생성하는 시스템

이 프로젝트는 Azure OpenAI API를 기반으로 작동하며, OpenAI의 최신 모델을 활용해 문맥에 맞는 응답을 제공합니다.

## 기능

1. LangGraph 기반 상태 관리: 멀티 에이전트 시스템에서 메시지와 상태를 효과적으로 관리합니다.
2. 다중 노드 및 병렬 처리: 에이전트 간 협업을 통해 복잡한 워크플로우를 구현합니다.
3. OpenAI API 통합: ChatGPT 및 임베딩 모델을 사용해 텍스트 생성 및 질의응답을 수행합니다.
4. 플로우 추적: 사용자의 요청이 각 에이전트를 거치는 과정을 시각적으로 디버깅할 수 있습니다.

추가적으로, 각 노드는 사용자가 정의한 상태와 입력을 처리하여 수월한 확장을 지원합니다.

## 설치 및 실행 방법

### 1. 환경 변수 설정
프로젝트를 실행하기 전에 아래 환경변수를 설정합니다. 이는 Azure OpenAI API 키 및 배포 ID를 포함합니다.

- `AOAI_ENDPOINT`: Azure OpenAI API의 엔드포인트
- `AOAI_API_KEY`: Azure OpenAI API의 접근 키
- `AOAI_DEPLOY_GPT4O`: GPT-4 모델 배포 이름
- `AOAI_DEPLOY_EMBED_ADA`: 임베딩 모델 배포 이름

환경 변수 설정 예시 (Linux/Mac):
```bash
export AOAI_ENDPOINT=
export AOAI_API_KEY=
export AOAI_DEPLOY_GPT4O="gpt-4o"
export AOAI_DEPLOY_GPT4O_MINI="gpt-4o-mini"
export AOAI_DEPLOY_EMBED_3_LARGE="text-embedding-3-large"
export AOAI_DEPLOY_EMBED_3_SMALL="text-embedding-3-small"
export AOAI_DEPLOY_EMBED_ADA="text-embedding-ada-002"

#### 5. 예제 코드
```markdown

## 사용 예제
아래는 이 프로젝트에서 활용되는 멀티 에이전트 시스템의 기본 사용 예제입니다.

```python
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

# 상태(TypeDict) 정의
class State(TypedDict):
    messages: list

# LangGraph 초기화
graph = StateGraph(State)

# 노드 정의
def sample_node(state: State) -> dict:
    response_text = f"Echo: {state['messages'][-1]['content']}"
    return {"messages": [AIMessage(content=response_text)]}

# 노드 추가
graph.add_node("sample_node", sample_node)
graph.set_entry_point("sample_node")
graph.set_finish_point("sample_node")

# 실행
output = graph.invoke({"messages": [HumanMessage(content="안녕하세요")]})
print(output["messages"][-1]["content"])

---

#### 6. 학습 자료 및 참고 링크
```markdown
## 학습 자료 및 참고 링크

- **LangChain 공식 문서**: [https://langchain.com/docs](https://langchain.com/docs)
- **LangGraph 사용법**: [https://langgraph.com/examples](https://langgraph.com/examples)
- **Azure OpenAI API**: [https://learn.microsoft.com/en-us/azure/openai](https://learn.microsoft.com/en-us/azure/openai)
- **GitHub Repository**(예시): [https://github.com/your-repo](https://github.com/your-repo)

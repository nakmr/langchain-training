from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

tools = load_tools(
    [
        "requests_all",
    ]
)

agent = initialize_agent(
    tools=tools, # Agentが使うツールを指定
    llm=chat, # Agentが使うLLMを指定
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # Agentのタイプを指定
    verbose=True
)

result = agent.run(
    """
    以下のURLにアクセスして東京の天気を調べて、日本語で答えてください。
    https://www.jma.go.jp/bosai/forecast/data/overview_forecast/130000.json
    """
)

print(f"実行結果: {result}")
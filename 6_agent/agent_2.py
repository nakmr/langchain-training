from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.tools.file_management import WriteFileTool

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
)

tools = load_tools(
    [
        "requests_get",
        "serpapi"
    ],
    # llm=chat,
)

tools.append(
    WriteFileTool(
        root_dir="./"
    )
)

agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    verbose=True
)

result = agent.run("大阪の名産品を調べて、日本語で、result.txtというファイルに書き込んでください。")
print(f"実行結果: {result}")
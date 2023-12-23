from calendar import c
from json import tool
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import WikipediaRetriever
from langchain.tools.file_management import WriteFileTool

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
)

tools = []

tools.append(
    WriteFileTool(
        root_dir="./"
    )
)

retriever = WikipediaRetriever(
    lang="ja",
    doc_content_chars_max=500,
    top_k_results=2
)

tools.append(
    create_retriever_tool(
        name="WikipediaRetriever",
        description="Retrieve the Wikipedia page of the specified keyword.",
        retriever=retriever
    )
)

agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("バーボンウイスキーの歴史を調べて、概要を日本語で、result.txtというファイルに保存してください。")
print(f"実行結果: {result}")
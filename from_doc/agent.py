from langchain.chat_models import ChatOpenAI
from langchain.agents import tool, AgentExecutor  
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.agents import AgentFinish
from langchain_core.messages import AIMessage, HumanMessage

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]

MEMORY_KEY = "chat_history"
chat_history = []
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words.",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        (
            "user",
            "{input}"
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind(
    functions=[
        format_tool_to_openai_function(t) for t in tools
    ]
)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

# result = agent.invoke(
#     {
#         "input": "How many letters in a word educa?",
#         "intermediate_steps": []
#     }
# )

# user_input = "How many letters in a word educa?"
# intermediate_steps = []

# while True:
#     output = agent.invoke(
#         {
#             "input": user_input,
#             "intermediate_steps": intermediate_steps
#         }
#     )

#     if isinstance(output, AgentFinish):
#         final_result = output.return_values["output"]
#         break
#     else:
#         print(f"TOOL NAME: {output.tool}")
#         print(f"TOOL INPUT: {output.tool_input}")
#         tool = {"get_word_length": get_word_length}[output.tool]
#         observation = tool.run(output.tool_input)
#         intermediate_steps.append((output, observation))

# print(f"最終結果: {final_result}")

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

input1 = "How many letters in a word educa?"
result = agent_executor.invoke(
    {
        "input": input1,
        "chat_history": chat_history
    }
)

chat_history.extend(
    [
        HumanMessage(content=input1),
        AIMessage(content=result["output"]),
    ]
)

result2 = agent_executor.invoke(
    {
        "input": "Is that a real word?",
        "chat_history": chat_history
    }
)

print(f"最終結果: {result2}")
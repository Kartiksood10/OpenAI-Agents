import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, LLMChain
#from htmlTemplates import css, bot_template, user_template
from langchain.memory import ChatMessageHistory
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain.tools import BaseTool
from math import pi
from typing import Union
import yfinance as yf

load_dotenv(find_dotenv(),override=True)

prompt = hub.pull("hwchase17/react-chat")

# input_variables = ['Calculator', 'Search', 'agent_scratchpad', 'chat_history', 'input', 'wikipedia', 'tools', 'tool_names']

# template = """Assistant is a large language model trained by OpenAI.

# # Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

# # Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

# # Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

# # TOOLS:
# # ------

# # Assistant has access to the following tools:

# # wikipedia: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.
# # Search: A wrapper around Google Search. Useful for when you need to answer questions about current events. Input should be a search query. 
# # Calculator: Useful for when you need to answer questions about math.

# # To use a tool, please use the following format:

# # ```
# # Thought: Do I need to use a tool? Yes
# # Action: the action to take, should be one of [wikipedia, Search, Calculator]
# # Action Input: the input to the action
# # Observation: the result of the action
# # ```

# # When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

# # ```
# # Thought: Do I need to use a tool? No
# # Final Answer: [your response here]
# # ```

# # Begin!

# # Previous conversation history:
# # {chat_history}

# # New input: {input}
# # {agent_scratchpad}
# # """

# input_variables=['agent_scratchpad', 'chat_history', 'input', 'tool_names', 'tools']
# template = """
# Answer the following questions as best you can. You have access to the following tools:

# wikipedia: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.
# Search: A wrapper around Google Search. Useful for when you need to answer questions about current events. Input should be a search query. 
# Calculator: Useful for when you need to answer questions about math.

# Use the following format:

# Question: the input question you must answer
# Thought: you should think step by step what actions you need to take
# Action: the action to take, should be one of [wikipedia, Search, Calculator]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!

# Question: {input}
# Previous conversation history:\n{chat_history}
# Thought:{agent_scratchpad}

# """

# template = """

# Answer the following questions as best you can. 

# Use the following format:

# Thought: Let's understand the user's question/problem better.
# Action: Clarification
# Action Input: {input}
# Observation: Parse the input to identify key components.

# Thought: Based on the parsed input, we need further details.
# Action: Clarification
# Action Input: [Specific clarifying question related to the parsed input]
# Observation: Gather more information from the user.

# ... (Repeat the above "Clarification" action and "Observation" step as needed for more details)

# Thought: I now have enough information to provide a recommendation.
# Action: Recommendation
# Action Input: [Data collected from the user]
# Observation: Generate the best answer or solution based on the collected information.

# Thought: Conversation complete. Providing the final answer to the user's question.
# Final Answer: [Final recommendation or solution]

# Question: {input}
# Previous conversation history:\n{chat_history}
# Thought: {agent_scratchpad}

# """
# prompt.input_variables = input_variables
# prompt.template = template
#print(prompt.template)

# CREATING CUSTOM TOOLS 

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

class CircumferenceTool(BaseTool):
    name = "Circumference calculator"
    description = "use this tool when you need to calculate a circumference using the radius of a circle"

    def _run(self, radius: Union[int, float]):
        return float(radius)*2.0*pi

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")
    
class FinanceTool(BaseTool):
    name = "yfinance"
    description = "use this tool when you need to answer questions about current stock price of a particular stock."

    def _run(self, query: str)-> str :
        tk = yf.Ticker(query)
        return tk.info.get("currentPrice")

    def _arun(self, query: str)->str:
        raise NotImplementedError("This tool does not support async")

search = SerpAPIWrapper()
wiki = WikipediaAPIWrapper()

# defining a single tool
search_tool = Tool(
        name = "search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    )

wiki_tool = Tool(
        name = "wiki",
        func=wiki.run,
        description="Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query."
    )


repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=search.run,
)

llm = OpenAI(temperature=0.5)

tools = [FinanceTool(), CircumferenceTool(), search_tool, wiki_tool, repl_tool, retriever_tool]

# DEFAULT TOOLS

#tools = load_tools(['wikipedia', 'serpapi','llm-math', 'dalle-image-generator', 'retriever_tool'], llm=llm)

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {
        "history": []
    }

def get_text():

    input_text = st.chat_input("Ask a question")
    return input_text

st.title("OpenAI Chatbot with Agents")
st.markdown("Welcome to My world!")

user_input = get_text()

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# menu = ["IT Industry", "Healthcare", "Insurance", "Education", "Automobile"]
# industry_choice = st.sidebar.selectbox("Select Industry", menu)
# # Enter client name
# client_name = st.sidebar.text_input("Enter Client Name", "")
# context_input = st.sidebar.text_input("Enter your Context")

# Process context when the user clicks the button
# if st.sidebar.button("Process"):
#     # backend code for Process the selected industry and client name
#     st.write(f"Processing Industry: {industry_choice}")
#     st.write(f"Processing Client Name: {client_name}")
#     st.write(f"Business Context,{context_input}")

if user_input:
    answer = agent_executor.invoke(
    {
        "input": user_input,
        "chat_history": st.session_state["chat_history"]
    }
)
    
    # output of react agent comes in the form of 'output':

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['output'])
    st.session_state.chat_history['history'].append({'user':user_input, 'output':answer['output']})
    print(st.session_state['chat_history'])

    for i in range(len(st.session_state['generated'])):
        st.info(st.session_state["past"][i])
        st.success(st.session_state['generated'][i], icon="🤖")
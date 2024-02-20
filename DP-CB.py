import streamlit as st
import pandas as pd
from openai import OpenAI
import openai
import re
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.agents import create_sql_agent
import pinecone
from dotenv import load_dotenv

# App title
st.set_page_config(initial_sidebar_state='collapsed')
st.header("Data Pilot GPT 💬:rocket:")
credentials = ["manaal1", "mojiz2", "mankee3", "zainab4", "eshaStop"]

input_credentials = st.sidebar.text_input("Please enter your valid PM Authorization key: ", type="password")
if not input_credentials in credentials:
    st.warning('Please enter your credentials on the side bar to the left top corner ">" to activate Project Manager Mode', icon='⚠️')
else:
    st.success('Proceed to being the fabulous PM that you are - or not', icon='👉')


load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENVIRONMENT')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client2 = OpenAI(api_key=OPENAI_API_KEY)
client3 = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=OPENAI_API_KEY, temperature=0, max_tokens = 150)
client4 = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=OPENAI_API_KEY, temperature=0, max_tokens = 500)
#llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0, streaming=True)
client = pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(host = "https://dp-index-057db86.svc.gcp-starter.pinecone.io")
model = 'text-embedding-ada-002'
openai.api_key = OPENAI_API_KEY

db_filepath = (Path(__file__).parent / "fashion_db.sqlite").absolute()
db_uri = f"sqlite:///{db_filepath}"

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

#prompts
delimiter = '####'

#layer-1
#1
query_classifier = f"""Classify user queries into three categories:

Task Nomenclature Generation:

User asks how to create a task name in ClickUp.
Example: "How do I name a task for a meeting with the marketing team?"

Company Policies and Documentation:

User asks about HR policies, ClickUp guidelines, or other company information found in the vector database.
Example: "What's the policy on working from home?"
Example: "Where would i log internal meeting on clickup?"
Example: "Where would i log the annual dinner on clickup?"

Project Management:

User asks about time spent on projects, hours logged by individuals, or other project-related data.
Example: "How many hours did we spend on the Blueconics project last week?"
Example: "Can you giive me a breakdown of the hours logged by team members on the clickup this week?"
Example: "Weekly progress Report"

Provide your output in string values: 'T' for Task nomenclature generation, 'C' for company policies and documentation and 'P' for project management.
"""
#2
task_nomenclature_gen = f"""Help me create clear task names based on the user's query. The user asks me how to name a task. I use the details from their query to create a name like:

[Action] - [Specific details]

Examples:

Meeting notes - Discuss onboarding plan with Sarah
Internal Meeting - Chatbot Discussion with Salman
Data validation - Blueconics dashboard Q4
Dashboard development - Update 7knots sales pipeline
Client Meeting - Weekly Progress Update
Client Communication - Email - Project Updates
Client Communication - Teams - Weekly Progress Update
Project Management - Weekly Progress Report
ML - Churn Prediction Model for Benzinga
Estimation - Revise Hours & Update Client
Data Scraping - POC
Self-Paced Study - AWS Concepts & Services
Team Training - Bi Weekly Session
HR Support - Skill Set Mapping

"""
#3
vec_db =  f"""You will be using the vector database to look up specific information. You will then use the information available to answer questions. \
Do not ask the user to consult the manual but use the entire information available to answer the questions on your own. \

Examples of questions:
I am a part time employee, what are my benefits?\
I am a part time employee so my benefits are different than a full time employee. \
I am a part time employee how many leaves do I get?\
How many leaves am I allowed in a year?\
Where would I log a certain task in clickup?

"""

#NLP-SQL Prompt
# Set up system message prompts for different scenarios

NLP_SQL = f""" You are a Database Expert where by you will be writing SQL queries from the prompts the user enters.\
 
Some examples of the situation are defined below\
 
 
User:
 
'ETL time taken in all space'\n
                                   '\n'
                                   'All spaces are consisting of different variables '
                                   'including space_name, folder_name, time spent '
                                   'as well as time estimate, assignee,creator, current status and name '
                                   'As a Database expert I will query the database '
                                   'and make the right call as to what is the answer.\n '
                                   'I will query the the mentioned terms while '
                                   'not mixing the answers.\n'
                                   
                                   
                                   
Assistant:
SQLQuery:SELECT SUM(time_spent) FROM tasks WHERE name LIKE '%ETL%' AND space_name IS NOT NULL
SQLResult: [(19.5,)]
Final Answer: 19.5 hours
 
User:
 
'The previous answer is incomplete, see the schema again and redo the query'\n
                                        '\n'
                                        'All spaces are consisting of different variables '
                                        'including space_name, folder_name, time spent '
                                        'as well as time estimate, assignee,creator, current status and name '
                                        'As a Database expert I will query the database '
                                        'and make the right call as to what is the answer.\n '
                                        'I will query the the mentioned terms while '
                                        'not mixing the answers.\n'
 
Assistant:
SQLQuery:
SQLResult:
Final Answer:  
               
 
Your job is the give your response in bullet format and to always quote numerical values instead of using words to define a range of a given time period.
 

"""
system_mes_temp = SystemMessagePromptTemplate.from_template(template= vec_db)
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_mes_temp, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory = st.session_state.buffer_memory, llm=client3, prompt=prompt_template, verbose= True)

@st.cache_resource(ttl="5h")
def configure_db(db_uri):
    try:  
        return SQLDatabase.from_uri(database_uri=db_uri)
    except Exception as e:
         print(f"Error occurred - DB not found: {e}")
         return None
    
db = configure_db(db_uri)

toolkit = SQLDatabaseToolkit(db=db, llm=client4)

agent = create_sql_agent(
    llm=client4,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True 
)

if 'token_usage' not in st.session_state:
    st.session_state['token_usage'] = []

#core-function
def get_completion_from_messages1(messages, model='gpt-3.5-turbo-16k', temperature=0.2, max_tokens=200):
    try:
        response = client2.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        response_message = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        return response_message, tokens_used
    except Exception as e:
         #Handle the exception
         print(f"An error occured in get_completion_from_messages: {e}")
         return None, None

def query_refiner(input):
    try:
        response = client2.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{'role':'assistant', 'content': vec_db},
                {'role':'user', 'content': f'{delimiter}{input}{delimiter}'}],
        temperature=0.1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        tokens_used = response.usage.total_tokens
        return response.choices[0].message.content, tokens_used
    
    except Exception as e:
         #haandle the exception
         print(f"An error occcured in query_refiner: {e}")
         return None, None

#logic-function for prompts
def get_meeting_keywords(input):
    try: 
        keywords = []
        meeting_words = re.findall(r"(meeting|agenda|discussion|sync|brainstoring|chat|call|conference)", input, re.IGNORECASE)
        keywords.extend(meeting_words)

        meeting_types = re.findall(r"(internal|one-on-one|team|client|review|status)", input, re.IGNORECASE)
        keywords.extend(meeting_types)

        topic_keywords = re.findall(r"(project|product|marketing|sales|finance|team|department)", input, re.IGNORECASE)
        keywords.extend(topic_keywords)

        return list(set(keywords))
    
    except Exception as e:
         print(f"An error occurred in finding matching keywords for meetings: {e}")
         return []

def get_sql_keywords(input):
    try:
        keywords2 = []
        pm_words = re.findall(r"(benzinga|7knots|estimated time|hours|hours spent|how much time does|how many hours)", input, re.IGNORECASE)
        keywords2.extend(pm_words)

        return list(set(keywords2))
    except Exception as e:
         print(f"An error occurred in getting SQL query keywords: {e}")
         return []
#task_nomenclature_gen.format(meeting_keywords = keywords)
#vectorDB
def find_match(input):
    try:
        xq = openai.embeddings.create(input=input, model=model).data[0].embedding
        result = index.query([xq], top_k=5, include_metadata=True)
        print(result)
        return result['matches'][0]['metadata']['text']+result['matches'][1]['metadata']['text']
    
    except Exception as e:
         print(f"An error occurred in find_match function: {e}")
         return None

#def has_meeting_keywords(input):

def input_classifier(input):
    try:
        messages = [{'role': 'assistant', 'content': query_classifier},
                {'role': 'user', 'content': f'{delimiter}{input}{delimiter}'}]
        response, tokens_used = get_completion_from_messages1(messages, max_tokens=1) 
        print(response)

        # Initialize token usage for the current response
        response_token_usage = []

        if tokens_used is not None:
            response_token_usage.append(tokens_used)
            st.session_state['token_usage'].append(response_token_usage)
        
        if response == 'T':
            history = st.session_state.messages
            keywords = get_meeting_keywords(input)
            messages =  [{'role':'assistant', 'content': task_nomenclature_gen.format(meeting_keywords = keywords)},
                {'role':'user', 'content': f'{delimiter}{input}{delimiter}'}]
            messages.extend(history)
            response2, tokens_used = get_completion_from_messages1(messages, max_tokens=500)
            
            if response2 is not None:
                st.session_state['responses'].append(response2)
            return response2

            # Store token usage for the current response
            response_token_usage = []
            if tokens_used is not None:
                response_token_usage.append(tokens_used)
                st.session_state['token_usage'].append(response_token_usage)
            st.subheader("Token Usage Information:")
            st.write('token_usage')

        elif response == 'C':
            history = st.session_state.messages
            #conversation_string = get_conversation_string()
            #refined_query = query_refiner(conversation_string, query)
            #messages = [{'role':'system', 'content': vec_db},
                    #{'role':'user', 'content': input}]
            response_1 = find_match(input)
            input_with_context = f"""
            Prompt:
            {vec_db}

            Context:
            {response_1}

            Query:
            {input}

            History:
            {history}
            """
            #messages =  [{'role':'assistant', 'content': vec_db},
               # {'role':'user', 'content': response_1}]
            response3 = conversation.predict(input =input_with_context)
            print(response3)
            if response3 is not None:
                st.session_state['responses'].append(response3)
            return response3

            # Store token usage for the current response
            response_token_usage = []
            if tokens_used is not None:
                response_token_usage.append(tokens_used)
                st.session_state['token_usage'].append(response_token_usage)

            # Display token usage information
            st.subheader("Token Usage Information:")
            st.write('token_usage')
        
        if response == 'P':
            # Get user input for authentication only for the 'P' case
                    # Proceed with the SQL-related functionality only after authentication
                    if input_credentials in credentials:
                        keywords = get_sql_keywords(input)
                        messages = [{'role':'system', 'content': NLP_SQL.format(pm_keywords=keywords)},
                                    {'role':'user', 'content': f'{delimiter}{input}{delimiter}'}]
                        response4 = str (agent.run(messages))
                        if response4 is not None:
                            st.session_state['responses'].append(response4)
                        return response4
                    else:
                        return "You are not authorized, please stop jumping out of the dataplane"
    except Exception as e:
         print(f"An error occured classifying the input as per the user query and the bot parameters: {e}")
         return "An error occurred, please refresh and try again", None


pdf = st.file_uploader("Upload your PDF", type='pdf')
underlying_embeddings = OpenAIEmbeddings()

store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model)
if pdf is not None:
        #pdf reader
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        db = FAISS.from_texts(chunks, cached_embedder)

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello jee. Let's start asking?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.button('Clear Chat History', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input("Write your query here "):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
else:
    if pdf is not None:
        prompt = st.text_input("Query the PDF now, click the x button on the side of the uploaded PDF to return to bot")
        docs = db.similarity_search(query=prompt, k=3)
        chain = load_qa_chain(llm=client4, chain_type="stuff")

        with get_openai_callback() as cb:
            bot_response = chain.run(input_documents=docs, question=prompt)
            st.write(bot_response)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Bro eik second de.."):
            response = str(input_classifier(prompt))
            placeholder = st.empty()
            full_response = ''
            for item in response:
                try:
                    full_response += item
                    placeholder.markdown(full_response)
                except:
                    pass
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

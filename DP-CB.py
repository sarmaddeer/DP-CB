import streamlit as st
import pandas as pd
from openai import OpenAI
import openai
import re
import os
import pickle
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from PyPDF2 import PdfReader
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
from InstructorEmbedding import INSTRUCTOR
from langchain.agents import create_sql_agent
import pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENVIRONMENT')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client2 = OpenAI(api_key=OPENAI_API_KEY)
client3 = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=OPENAI_API_KEY, temperature=0)
#llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0, streaming=True)
client = pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index('dp-index')
model = 'text-embedding-ada-002'
openai.api_key = OPENAI_API_KEY

db_filepath = (Path(__file__).parent / "fashion_db.sqlite").absolute()
db_uri = f"sqlite:///{db_filepath}"


if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)
memory = st.session_state.buffer_memory

if st.button("Clear message history"):
    st.session_state['responses'] = ["How can I assist you? You can ask me about the task nomenclature or even the HR policies"]


if "messages" not in st.session_state:
    st.session_state.messages = []

#prompts
delimiter = '####'

#layer-1
#1
query_classifier = f"""You are a helpful assistant who's task is to classify the user queries according to the input into three separate nodes. The user query will present itself in a question. \
The user can ask you to either generate a task name and nomenclature for logging it on clickup based on the query. Or the user can ask you questions about task nomenclature guidelines and HR policies in the documentation found \
on the vector database. Or the user can ask you questions based on project management. This can include questions relating to the time spent on a particular project or the number of hours spent by someone. \
The given user input will be provided between delimiters i.e., {delimiter}. \

Your task is to classify the whether the query would be related to generating task nomenclature, document referencing of nomenclature guidelines and HR policies or whether it is a project management query. \
Mostly generating task nomenclature would be from the user stating their task. User queries about task nomenclature guidelines would be asking about how to name certain task types based on the clickup guidelines. \
User queries based on project management would be time related or relating to a particular person or project. \

Examples of task nomenclature: Task about internal meetings, How to write about dashboard discussion on blueconic dashboards, devops task nomenclature examples, etc. \
Examples of company policies: How many medical leaves do I get, What is the procedure to apply for leaves, What is my medical allowance, Company policies on Clickup SOPs, Daily Standup information, etc. \
Examples of project management: How many hours did Sarmad spend on chatbot yesterday, how many hours were spent on blueconic last week, how many people worked on 7knots yesterday and for how long, etc. \

Provide your output in string values: 'T' for Task nomenclature, 'C' for company policies and 'P' for project management.
"""
#2
task_nomenclature_gen = f"""you are a helpful assistant who's task is to generative task nomenclature based on the examples provided. \
The given user input will be provided between delimiters i.e., {delimiter}. \
Provide the proper task nomenclature as defined in the examples below: \

the user can ask you query about the how to name a task which you will use the information in the query to type it out like "[Task type] - [Description of the task].\

Example of prompts would include the user asking: "{delimiter}What is the task nomenclature for meetings{delimiter}"\
AI: "can you tell me what kind of a meeting it is"\
User: "{delimiter}I have a meeting with my teammate Ali where I ask him how to work on the customer churn prediction model{delimiter}"\
AI: "[Internal Meeting] - [Discussion with Ali on Customer Churn Prediciton Model] \

Another example could be a user asking: "{delimiter}what would be the task nomenclature for data validation for blueconics dashboard{delimiter}" \
AI: "Log this task in the Blueconics list under the appropriate space and then log in the task as follows: Data Validation - Blueconics Dashboard" \

Another Example could be a user asking: "{delimiter}What would be the task nomenclature for working on dashboards for 7knots{delimiter}" \
AI: "Dashboard Development - Dashboard Update for 7knots" \

"""
#3
vec_db =  f"""You are an assistant who helps the user with information from the available database. A query text will be provided usually in the form of a question and your job will be to provide a response based on the chat memory {memory} and the {input}. \
You will make sure that the response is concise and to the point.\
All you responses will be in a bulleted list and you will not exceed more than 4 points in your responses. \

the query will be asking you questions about clickup task nomenclature which you can look up referenced from the available database. The response will be short and to the point, \
lastly the user can ask you about company policies which you are to look in from the available database and revert with an answer as consice as possible. \

Examples of clickup task nomenclature query: "How do we write meetings on clickup". \
Example of policies query: "Can you tell me about employee leaves". \
Example of AI response: "Employee leaves are as follows\
1 - Casual leaves - 8\
2 - Annual leaves - 10\
3 - Mental Health - 2" \

"""

#NLP-SQL Prompt
# Set up system message prompts for different scenarios

NLP_SQL = f""" You are a Database Expert where by you will be writing SQL queries from the given prompt. The examples of your tasks is as follows where you're the Database Expert and the user will be the Project Manager:\

Project Manager: Obtain a high-level overview of the ongoing projects. Retrieve details such as project names, descriptions, status, and assignees.\

Database Expert: Craft SQL queries to extract information from the columns id, name, description, current_status, and assignee. Ensure the responses are clear and concise for the Project Manager to quickly understand the project landscape.\

Project Manager: Track the progress of selected projects. Retrieve information on dates (created, closed), time estimates, and time spent for effective project management.\

Database Expert: Create SQL queries to extract details from columns date_created, date_closed, time_estimate, and time_spent. Provide insights into the temporal aspects of the projects for better tracking and analysis.\

Project Manager: Focus on project assignments and due dates. Retrieve information about assignees, due dates, and identify any overdue tasks that require attention.\

Database Expert: Construct SQL queries to fetch data from columns assignee, due_date, and current_status. Highlight any projects with approaching or overdue deadlines for the Project Manager to address promptly.\

Project Manager: Explore collaboration within project teams. Retrieve data on team members, their roles, and collaborative efforts.\

Database Expert: Formulate SQL queries to extract information from columns assignee, creator, and current_status. Analyze team dynamics and provide insights into how team members are contributing to project success.\

Project Manager: Gain insights into client interactions and satisfaction. Retrieve data on client interactions, feedback, and project success metrics.\

Database Expert: Create SQL queries to extract information from columns id, name, description, and client-related columns. Provide a comprehensive view of client interactions and satisfaction levels for the Project Manager's analysis.\

Your job is the give your response in bullet format and to always quote numerical values instead of using words to define a range of a given time period.

"""

# Add a dictionary to store valid usernames and passwords
authorized_users = {'manaal1': 'hello123', 'mojiz3': 'hello456'}  # Add your authorized users

# Function to check user authentication
def authenticate(username, password):
    return username in authorized_users and password == authorized_users[username]

@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    return SQLDatabase.from_uri(database_uri=db_uri)

db = configure_db(db_uri)

toolkit = SQLDatabaseToolkit(db=db, llm=client3)

agent = create_sql_agent(
    llm=client3,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

if 'token_usage' not in st.session_state:
    st.session_state['token_usage'] = []

#core-function
def get_completion_from_messages1(messages, model='gpt-3.5-turbo', temperature=0.2, max_tokens=500):
    response = client2.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    response_message = response.choices[0].message.content
    tokens_used = response.usage.total_tokens
    return response_message, tokens_used

def query_refiner(input):

    response = client2.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{'role':'system', 'content': vec_db},
              {'role':'user', 'content': f'{delimiter}{input}{delimiter}'}],
    temperature=0.1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    tokens_used = response.usage.total_tokens
    return response.choices[0].message.content, tokens_used

credentials = ['090078601']
#logic-function for prompts
def get_meeting_keywords(input):
  keywords = []
  meeting_words = re.findall(r"(meeting|agenda|discussion|sync|brainstoring|chat|call|conference)", input, re.IGNORECASE)
  keywords.extend(meeting_words)

  meeting_types = re.findall(r"(internal|one-on-one|team|client|review|status)", input, re.IGNORECASE)
  keywords.extend(meeting_types)

  topic_keywords = re.findall(r"(project|product|marketing|sales|finance|team|department)", input, re.IGNORECASE)
  keywords.extend(topic_keywords)

  return list(set(keywords))
  pass

def get_sql_keywords(input):
    keywords2 = []
    pm_words = re.findall(r"(benzinga|7knots|estimated time|hours|hours spent|how much time does|how many hours)", input, re.IGNORECASE)
    keywords2.extend(pm_words)

    return list(set(keywords2))
    pass
#task_nomenclature_gen.format(meeting_keywords = keywords)
#vectorDB
def find_match(input):
    xq = openai.embeddings.create(input=input, model=model).data[0].embedding
    result = index.query([xq], top_k=5, include_metadata=True)
    print(result)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']
   

#def has_meeting_keywords(input):

def input_classifier(input):
    messages = [{'role': 'system', 'content': query_classifier},
            {'role': 'user', 'content': f'{delimiter}{input}{delimiter}'}]
    response, tokens_used = get_completion_from_messages1(messages, max_tokens=1) 
    print(response)

    # Initialize token usage for the current response
    response_token_usage = []

    if tokens_used is not None:
        response_token_usage.append(tokens_used)
        st.session_state['token_usage'].append(response_token_usage)
    
    if response == 'T':
        keywords = get_meeting_keywords(input)
        messages =  [{'role':'system', 'content': task_nomenclature_gen.format(meeting_keywords = keywords)},
              {'role':'user', 'content': f'{delimiter}{input}{delimiter}'}]
        response2, tokens_used = get_completion_from_messages1(messages, max_tokens=500)
        print(response2)
        st.session_state['responses'].append(response2)

        # Store token usage for the current response
        response_token_usage = []
        if tokens_used is not None:
            response_token_usage.append(tokens_used)
            st.session_state['token_usage'].append(response_token_usage)
        st.subheader("Token Usage Information:")
        st.write('token_usage')

    elif response == 'C':
        #conversation_string = get_conversation_string()
        #refined_query = query_refiner(conversation_string, query)
        #messages = [{'role':'system', 'content': vec_db},
                #{'role':'user', 'content': input}]
        context, tokens_used = query_refiner(input) 
        response3 = find_match(context) 
        st.session_state['responses'].append(response3)

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
        auth_username = st.text_input("Enter your username:")
        auth_password = st.text_input("Enter your password:", type="password")

        # Check if the entered username and password are valid
        if st.button("Authenticate"):
            if authenticate(auth_username, auth_password):
                st.success("Authentication successful!")

                # Proceed with the SQL-related functionality only after authentication
                keywords = get_sql_keywords(input)
                messages = [{'role':'system', 'content': NLP_SQL.format(pm_keywords=keywords)},
                            {'role':'user', 'content': f'{delimiter}{input}{delimiter}'}]
                response4 = agent.run(messages)
                st.session_state['responses'].append(response4)
            else:
                st.error("Authentication failed. Access denied.")


        
    
st.header("Data Pilot GPT :rocket:")

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
    st.session_state['responses'] = ["How can I assist you? You can ask me about the task nomenclature or even the HR policies"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

#bot front
user_input = st.text_input("Query: ", key="input")
if st.button("Submit Query"):
    if pdf is not None:
        docs = db.similarity_search(query=user_input, k=3)
        chain = load_qa_chain(llm=client3, chain_type="stuff")

        with get_openai_callback() as cb:
            bot_response = chain.run(input_documents=docs, question=user_input)
            st.write(bot_response)
    else:
        if user_input:
            # Process user input and update session state
            #st.session_state.buffer_memory.add_user_message(user_input)
            context = input_classifier(user_input)
            bot_response = context
            st.session_state['requests'].append(user_input)
            st.session_state['responses'].append(bot_response)

# Display chat history
if st.session_state['requests']:
    for i in range(len(st.session_state['requests']) - 1, -1, -1):
        # Display user input
        message(st.session_state["requests"][i], is_user=True, key=str(i))
        # Display bot response
        message(st.session_state['responses'][i])


import numpy as np
import streamlit as st
import sqlite3
import os
import pandas as pd
import requests
from chat_handle import ChatHandler
import json

header_sticky = """
    <style>
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 2.875rem;
            background-color: white;
            z-index: 999;
        }
        .fixed-header { 
        }
    </style>
        """

def display_existing_messages_and_return_context():
    context_chat = ''
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    for message in st.session_state["messages"][-min(len(st.session_state["messages"]), 6):]:
        role = message["role"]
        content = message["content"]
        context_chat = context_chat + f"{role} : {content} |"
    return context_chat

def add_user_message_to_session(prompt):
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

def add_bot_message_to_session(response):
    if response:
        st.session_state["messages"].append({"role": "ai", "content": response})
        with st.chat_message("ai"):
            st.markdown(response)

def call_api(context_chat, prompt):
    handler = st.session_state['handler']
    tables = st.session_state['tables'][st.session_state['db_id']]
    result = handler.handle_chat(context_chat, tables, prompt)
    return result

@st.cache_resource 
def prepare_model():
    return ChatHandler()

@st.cache_data
def prepare_tables():

    with open("data/cosql/tables.json") as f:
        tmp = json.load(f)

    my_tables = {}
    for i in tmp:
        my_tables[i["db_id"]] = i 
    
    return my_tables

def chat():
    chat_container = st.container()
    if st.button("New conversation"):
        st.session_state["messages"] = []
        st.session_state["handler"].new_conversation()
        display_existing_messages_and_return_context()
    chat_container.header("ConverSQL-UIT", divider="rainbow")
    
    st.session_state['handler'] = prepare_model()   
    st.session_state['tables'] = prepare_tables()

    ### Custom CSS for the sticky header
    chat_container.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
    st.markdown(header_sticky, unsafe_allow_html=True)
    
    
    context_chat = display_existing_messages_and_return_context()
    query = st.chat_input("Say something")
    if query:
        add_user_message_to_session(query)
        response = call_api(context_chat, query)
        add_bot_message_to_session(response)
      

def sidebar():
    with st.sidebar:

        st.header('**Database**', divider='rainbow')
        list_db = os.listdir('data/database/')   
        db = st.selectbox('**Select Database**', list_db)
        st.session_state['db_id'] = db
        
        conn = sqlite3.connect(f'data/database/{db}/{db}.sqlite')
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        list_table = [i[0] for i in cursor.fetchall()]

        with st.expander('Tables'):
            df = pd.DataFrame(list_table, columns=['Table Name'])
            st.write(df)  

        with st.expander('Content'):
            choose_table = st.selectbox("Select table to preview content", list_table)
            sql_query = f"select * from {choose_table}"
            cursor.execute(sql_query)
            list_data = cursor.fetchall()
            columns = next(zip(*cursor.description))
            df = pd.DataFrame(list_data, columns=columns)
            st.write(df)

        st.header('**SQL query**', divider='rainbow')

        if not "current_sql" in st.session_state:
            st.session_state["current_sql"] = ""
        current_sql = st.session_state["current_sql"]

        if st.button("Execute last chatbot's response"):
            current_sql = st.session_state['messages'][-1]['content']
            if not "```sql" in current_sql:
                current_sql = ""
            else:
                while current_sql != "" and current_sql[0] != '`':
                    current_sql = current_sql[1:]
                current_sql = current_sql.replace("```", "")
                current_sql = current_sql.replace("sql", "", 1)
                st.session_state["current_sql"] = current_sql
        try:
            sql_query = st.text_area('**Enter Query**', current_sql)
            st.session_state["current_sql"] = sql_query
            cursor.execute(sql_query)
            list_data = cursor.fetchall()
            columns = next(zip(*cursor.description))
            df = pd.DataFrame(list_data, columns=columns)
            st.write(df)
        except:
            # write red color markdown
            if sql_query != "":
                st.markdown('<p style="color:red;">🙉 Error sql query!!!</p>', unsafe_allow_html=True)
    conn.close()
    
if __name__ == "__main__":
    chat()
    sidebar()


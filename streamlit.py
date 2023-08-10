import os 
from apikey import apikey 


import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import  PromptTemplate
from langchain.chains import LLMChain ,SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

OPENAI_API_KEY=apikey
os.environ['OPENAI_API_KEY']=OPENAI_API_KEY


#APP framework 
st.title('🦜️🔗 youtube GPT creator')

prompt= st.text_input('PLug in your prompt here')

title_template=PromptTemplate(
    input_variables=['topic'],
    template="write me a youtube vedio title about {topic}"
    
)
script_template=PromptTemplate(
    input_variables=['title'],
    template='write me a youtube video script based on this TITLE: {title} '
    
)
#memory 
title_memory= ConversationBufferMemory(input_key='topic',memory_key='chat_history')
script_memory= ConversationBufferMemory(input_key='title',memory_key='chat_history')

#LLMS 

llm= OpenAI()
title_chain=LLMChain(llm=llm,prompt=title_template,output_key='title',memory=title_memory)
script_chain=LLMChain(llm=llm,prompt=script_template,output_key='script',memory=script_memory)
#sequential_chain=SequentialChain(chains=[title_chain,script_chain],input_variables=['topic'],output_variables=['title','script'])
wiki=WikipediaAPIWrapper()

# SHOW stuff to the screen is prompt is available 
if prompt:
    title=title_chain.run(prompt)
    wiki_research=wiki.run(prompt)
    script=script_chain.run(title)
    st.write(title) 
    st.write(script) 
    
    with st.expander('Title History'):
        st.info(title_memory.buffer)
    with st.expander('Script History'):
        st.info(script_memory.buffer)
    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
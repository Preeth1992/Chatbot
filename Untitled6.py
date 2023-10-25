#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install langchain')
get_ipython().system('pip install openai')
get_ipython().system('pip install PyPDF2')
get_ipython().system('pip install faiss-cpu')
get_ipython().system('pip install tiktoken')


# In[2]:


from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


# In[14]:


import os
os.environ["OPENAI_API_KEY"] = "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"


# In[3]:


# provide the path of  pdf file/files.
pdfreader = PdfReader('E:/data science/book on ML/StatisticsMachineLearningPythonDraft.pdf')


# In[4]:


from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content


# In[5]:


raw_text


# In[6]:


# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


# In[7]:


len(texts)


# In[15]:


# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()


# In[16]:


document_search = FAISS.from_texts(texts, embeddings)


# In[17]:


document_search


# In[18]:


from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


# In[19]:


chain = load_qa_chain(OpenAI(), chain_type="stuff")


# In[21]:


query = "Covariance matrix"
docs = document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)


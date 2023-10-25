
get_ipython().system('pip install langchain')
get_ipython().system('pip install openai')
get_ipython().system('pip install PyPDF2')
get_ipython().system('pip install faiss-cpu')
get_ipython().system('pip install tiktoken')


from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


import os
os.environ["OPENAI_API_KEY"] = "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"


# provide the path of  pdf file/files.
pdfreader = PdfReader('E:/data science/book on ML/StatisticsMachineLearningPythonDraft.pdf')


from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content



raw_text


# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


len(texts)


# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()


document_search = FAISS.from_texts(texts, embeddings)

document_search


from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


chain = load_qa_chain(OpenAI(), chain_type="stuff")


query = "Covariance matrix"
docs = document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)


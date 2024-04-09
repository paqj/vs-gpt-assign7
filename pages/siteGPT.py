import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.storage import LocalFileStore

from tempfile import gettempdir
import os
from urllib.parse import urlparse


llm = ChatOpenAI(
    temperature=0.1,
)

# GPT가 특정 Site를 크롤링하고 그 정보로 알려줌.
# 1. playwright, chromimum
# 2. site loader

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

def parse_page(soup):
    # soup에서 header, footer 제거
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    # 공백, 줄바꿈, 반복문구 삭제
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )

# Streaming
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token



# Memory
memory = ConversationBufferMemory(
    llm=llm,
    max_token_limit=50,
    return_messages=True,
)

# 임시 디렉토리 경로 사용 예시
temp_dir = gettempdir()


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    # Domain 추출
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.split('.')

    if len(domain) > 2:
        domain = domain[-2]
    else:
        domain = domain[0]

    cache_base_dir = os.path.join(temp_dir, "streamlit_cache", domain)
    cache_dir = LocalFileStore(cache_base_dir)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    # html -> text 까지
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    # 2초 딜레이
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    # embedding
    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever

# For Streaming
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

# For History
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

url = None

with st.sidebar:
    user_api_key = st.text_input("Please enter your API key on app page")

    if user_api_key:
        st.session_state['api_key'] = user_api_key

        url = st.text_input("Write down a URL", placeholder="https://example.com")

        user_api_key = st.session_state['api_key']
        # API 키가 있는 경우 ChatOpenAI 객체에 적용
        llm = ChatOpenAI(
            api_key=user_api_key,  # API 키 적용
            temperature=0.1,
            streaming=True,
            callbacks=[ChatCallbackHandler()],
        )
        # Memory
        memory = ConversationBufferMemory(
            llm=llm,
            max_token_limit=50,
            return_messages=True,
        )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)

        send_message("I'm ready! Ask away!", "ai", save=False)

        paint_history()

        # query -> chain -> retriever : docs -> get_answer -> choose_answer
        query = st.chat_input("Ask a question to the website.")
        if query:
            send_message(query, "human")

            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            with st.chat_message("ai"):
                result = chain.invoke(query)
                st.markdown(result.content.replace("$", "\$"))
else:
    st.session_state["messages"] = []

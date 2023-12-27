from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import dotenv_values
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="ChowisGPT",
    description="chowis FAQ",
)


class GPTMessage(BaseModel):
    ok: bool = Field(description="Check internal process")
    message: str = Field(description="Message from GPT")


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        print("start")

    def on_llm_end(self, *args, **kwargs):
        print("end", self.message)


@app.post(
    "/gpt/faq",
    summary="Returns a message from GPT",
    description="User send retriever_file_name and user message and receive GPT answers",
    response_description="A GPTMessage object that contains the api working status and message from GPT",
    response_model=GPTMessage,
)
def ChowisGPT_FAQ(
    retriever_file_name="chowis_faq.txt",
    user_message=None,
):
    env_vars = dotenv_values(".env")

    if not retriever_file_name:
        return {
            "ok": False,
            "message": "Please give me a retriever_file_name",
        }

    if not user_message:
        return {
            "ok": False,
            "message": "Please give me a user_message",
        }

    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        openai_api_key=env_vars["OPENAI_API_KEY"],
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Answer the question using ONLY the following context. If you don't know the answer just say you don't know and GIVE this official website URL ("https://www.chowis.com/ko") as answer. DON'T make anything up. If there is supportive document given, answer with it. 
        
                Context: {context}
                """,
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    file_path = f".cache/files/{retriever_file_name}"
    cache_dir = LocalFileStore(f".cache/embeddings/{retriever_file_name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        openai_api_key=env_vars["OPENAI_API_KEY"],
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    response = chain.invoke(user_message)

    return {
        "ok": True,
        "message": response.content,
    }

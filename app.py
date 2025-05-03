import gradio as gr
import cassio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from src.agents.sql import sql_agent
from src.agents.rag import retrieve
from langgraph.graph import StateGraph, START, END
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal, Any
from typing_extensions import TypedDict
from src.gradiocallback import GradioCallbackHandler
from src.helper import process_pdfs
from src.prompt import route_system

# Load environment variables
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID_MULTI_AGENT = os.getenv("ASTRA_DB_ID_MULTI_AGENT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASS = os.getenv("MYSQL_PASS")
MYSQL_DB = os.getenv("MYSQL_DB")

ASTRA_KEYSPACE = os.getenv("ASTRA_KEYSPACE")
ASTRA_TBL = os.getenv("ASTRA_TBL")

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID_MULTI_AGENT)

class RouteQuery(BaseModel):
    datasource: Literal["retrieve", "sql_agent"] = Field(...)

session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

query_limit = 100

class GraphState(TypedDict):
    question: str
    llm: Any
    question_route: Any
    dbconfig: dict
    astraConfig: dict
    generation: str
    callbacks: Any
    documents: List[str]
    pdf_documents: Any
    get_session_history: Any
    session_id: Any
    agents: str
    result: str

def route_question(state):
    print("--- ROUTE QUESTION ---")
    question = state["question"]
    print(f"question: {question}")
    question_route_chain = state["question_route"]
    try:
        route_result = question_route_chain.invoke({"message_history": question})
        print(f"route_result: {route_result}")
        if isinstance(route_result, RouteQuery):
            return route_result.datasource
        elif isinstance(route_result, dict) and "datasource" in route_result:
            return route_result["datasource"]
    except Exception as e:
        print(f"Error invoking route chain: {e}")
    print("Warning: unexpected route format, defaulting to 'retrieve'")
    return "retrieve"

def check_result(state):
    return "success" if state["result"] else "fail"

pdf_documents = []

def chat_with_ai(message_history, question):
    try:
        model = "gpt-4o"
        api_key = OPENAI_API_KEY
        agent = "RAG-PDFs"

        if not api_key:
            yield "API Key is missing. Please configure the correct API key."
            return

        try:
            llm = ChatOpenAI(api_key=api_key, model=model, temperature=0, streaming=True)
            llm_route = llm.with_structured_output(RouteQuery)
        except Exception as e:
            yield f"Error initializing LLM: {str(e)}"
            return

        try:
            workflow = StateGraph(GraphState)
            workflow.add_node("sql_agent", sql_agent)
            workflow.add_node("retrieve", retrieve)
            workflow.add_conditional_edges(START, route_question, {
                "sql_agent": "sql_agent",
                "retrieve": "retrieve"
            })
            workflow.add_conditional_edges("sql_agent", check_result, {
                "success": END,
                "fail": "retrieve"
            })
            workflow.add_edge("retrieve", END)
            app = workflow.compile()
        except Exception as e:
            yield f"Error setting up workflow: {str(e)}"
            return

        question_routes = (
            ChatPromptTemplate.from_messages([
                ("system", route_system),
                ("human", "{message_history}")
            ]) | llm_route
        )

        inputs = {
            "question": message_history,
            "llm": llm,
            "question_route": question_routes,
            "dbconfig": {
                "host": MYSQL_HOST,
                "user": MYSQL_USER,
                "pass": MYSQL_PASS,
                "db_name": MYSQL_DB,
                "limit": query_limit,
            },
            "astraConfig": {
                "keyspace": ASTRA_KEYSPACE,
                "table": ASTRA_TBL
            },
            "callbacks": GradioCallbackHandler(gr.update),
            "pdf_documents": pdf_documents,
            "get_session_history": get_session_history,
            "session_id": "default_session",
            "agents": agent,
            "result": "fail"
        }

        for output in app.stream(inputs):
            if not isinstance(output, dict):
                yield f"Unexpected response format: {output}"
                return
            for key, value in output.items():
                if isinstance(value, dict) and "documents" in value and hasattr(value["documents"], "page_content"):
                    yield value["documents"].page_content + "\n"
                else:
                    if key == "retrieve":
                        yield "Searching in VectorStore..."
                    else:
                        yield "Error: Missing 'documents' in response."

        inputs["result"] = "success"

    except Exception as e:
        yield f"Unexpected error: {str(e)}"

with gr.Blocks() as app:
    gr.Markdown("# Ask AI")

    file_upload = gr.Files(file_types=[".pdf"], label="Upload PDFs")
    output_text = gr.Textbox(label="Status")
    file_upload.change(process_pdfs, inputs=file_upload, outputs=output_text)

    gr.ChatInterface(chat_with_ai, type="messages")

if __name__ == "__main__":
    app.launch()

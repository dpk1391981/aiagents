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
from src.agents.wikipedia import wiki_search
from langgraph.graph import StateGraph, START, END
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal, Any
from typing_extensions import TypedDict
from src.gradiocallback import GradioCallbackHandler
from src.helper import process_pdfs
from src.prompt import route_system
from PIL import Image
import io
import base64
from graphviz import Digraph

# Load environment variables
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID_MULTI_AGENT = os.getenv("ASTRA_DB_ID_MULTI_AGENT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MySQL setup
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASS = os.getenv("MYSQL_PASS")
MYSQL_DB = os.getenv("MYSQL_DB")

# Astra DB config
ASTRA_KEYSPACE = os.getenv("ASTRA_KEYSPACE")
ASTRA_TBL = os.getenv("ASTRA_TBL")

# Initialize Cassandra/AstraDB
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID_MULTI_AGENT)

# Data Model
class RouteQuery(BaseModel):
    datasource: Literal["retrieve", "sql_agent", "wiki_search"] = Field(
        ..., description="Route to SQL agent, retrieve, or Wikipedia."
    )

# Manage chat history
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

# Function to route questions
def route_question(state):
    print("--- ROUTE QUESTION ---")
    question = state["question"].lower()
    print(f"question: {question}")
    
    source = state["question_route"].invoke({"message_history": question})
    
    # Ensure we return a string instead of RouteQuery object
    if isinstance(source, RouteQuery):
        selected_source = source.datasource
    else:
        selected_source = "wiki_search"  # Default to Wikipedia if the response is unexpected
    
    print(f"LLM selected source: {selected_source}")
    return selected_source
    
def check_result(state):
    return "success" if state["result"] else "fail"


pdf_documents = []

# Main AI Chat Function

def chat_with_ai(message_history, question, api_key_type, agents):
    try:
        model = "gpt-4o" if api_key_type == "Open API" else "deepseek-r1-distill-llama-70b"
        api_key = OPENAI_API_KEY if api_key_type == "Open API" else GROQ_API_KEY

        if not api_key:
            yield "API Key is missing. Please configure the correct API key."
            return

        try:
            llm = (
                ChatOpenAI(api_key=api_key, model=model, temperature=0, streaming=True)
                if api_key_type == "Open API"
                else ChatGroq(groq_api_key=api_key, model=model, streaming=True)
            )
            llm_route = llm.with_structured_output(RouteQuery)
        except Exception as e:
            yield f"Error initializing LLM: {str(e)}"
            return

        try:
            workflow = StateGraph(GraphState)

            # Define nodes
            workflow.add_node("sql_agent", sql_agent)
            workflow.add_node("retrieve", retrieve)
            workflow.add_node("wiki_search", wiki_search)

            # Routing logic
            workflow.add_conditional_edges(
                START,
                route_question,
                {
                    "sql_agent": "sql_agent",
                    "retrieve": "retrieve",
                    "wiki_search": "wiki_search",
                }
            )

            # If SQL fails, try VectorStore
            workflow.add_conditional_edges("sql_agent", check_result, {"success": END, "fail": "retrieve"})

            # If VectorStore fails, try Wikipedia
            workflow.add_conditional_edges("retrieve", check_result, {"success": END, "fail": "wiki_search"})

            # Wikipedia is the last fallback
            workflow.add_edge("wiki_search", END)

            app = workflow.compile()

        except Exception as e:
            yield f"Error setting up workflow: {str(e)}"
            return

        question_routes = ChatPromptTemplate.from_messages([
            ("system", route_system),
            ("human", "{message_history}")
        ]) | llm_route

        inputs = {
            "question": message_history,  # Use question instead of message_history
            "llm": question_routes,
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
            "agents": agents,
            "result": "fail"
        }

        try:
            for output in app.stream(inputs):
                if not isinstance(output, dict):
                    yield f"Unexpected response format: {output}"
                    return

                for key, value in output.items():
                    if "documents" in value and hasattr(value["documents"], "page_content"):
                        chunk = value["documents"].page_content + "\n"
                        yield chunk
                    else:
                        if key == "retrieve":
                            yield "Searching in VectorStore..."
                        elif key == "wiki_search":
                            yield "Final search in Wikipedia..."
                        else:
                            yield "Error: Missing 'documents' in response."

        except Exception as e:
            yield f"Error during response generation: {str(e)}"

        inputs["result"] = "success"

    except Exception as e:
        yield f"Unexpected error: {str(e)}"


def toggle_upload(agent):
    return gr.update(visible=(agent == "RAG-PDFs"))

with gr.Blocks() as app:
    gr.Markdown("# Ask AI")

    with gr.Accordion("Additional Input", open=False):
        api_key_type = gr.Dropdown(["Open API", "Deepseek Ollama API"], label="Select LLM API")
        agents = gr.Dropdown(["RAG-PDFs", "SQL", "Wikipedia"], label="Select Agent")

    file_upload = gr.Files(file_types=[".pdf"], label="Upload PDFs")
    output_text = gr.Textbox(label="Status")
    file_upload.change(process_pdfs, inputs=file_upload, outputs=output_text)
    agents.change(toggle_upload, inputs=agents, outputs=file_upload)

    gr.ChatInterface(chat_with_ai, type="messages", additional_inputs=[api_key_type, agents])

if __name__ == "__main__":
    app.launch()

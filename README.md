# Gen AI App (Multi AI Agents) - Proof of Concept (POC)

## Overview
This project is a Gen AI application that integrates multiple AI agents to perform various tasks. The application consists of three key AI agents:

1. **PDF AI Agent** - Uploads a PDF and generates AI answers based on its content.
2. **SQL Query Agent** - Allows querying a database using natural language with an SQL tool.
3. **Wikipedia Agent** - Fetches relevant information from Wikipedia using AI.

## Tech Stack Used
- **Language:** Python
- **Gen AI Frameworks:** LangChain & LangGraph (Stateful)
- **Monitoring Tools:** LangSmith (LangOps)
- **Embeddings:** HuggingFace
- **Vector DB:** Cassandra DB - Astra DB
- **Prompt Engineering**
- **UI Framework:** Gradio
- **SQL Tools:** SQL
- **AI APIs:** OpenAI, Groq

## Installation Guide
Follow these steps to set up and run the application locally:

### Prerequisites
- Install Python on your system (Recommended version: Python 3.8+)
- Ensure `pip` is installed and up to date

### Steps to Install and Run
1. **Clone the Repository**
   ```bash
   git clone https://github.com/dpk1391981/aiagents
   ```
2. **Navigate to the Project Directory**
   ```bash
   cd aiagents
   ```
3. **Create a Virtual Environment**
   - **For Linux/macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **For Windows (CMD):**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **Alternatively, using Conda:**
     ```bash
     conda create --name genai_env python=3.8
     conda activate genai_env
     ```
4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
5. **Set Up the Database**
   - Run the following SQL file in your system to set up the required database:
     ```bash
     database/at_clothes.sql
     ```
6. **Set Up Environment Variables**
   - Create a `.env` file and add the following (replace placeholders with actual values):
     ```ini
     HF_TOKEN=<your_huggingface_token>
     GROQ_API_KEY=<your_groq_api_key>
     ASTRA_DB_ID=<your_astra_db_id>
     OPENAI_API_KEY=<your_openai_api_key>
     MYSQL_HOST="localhost"
     MYSQL_USER="your_mysql_user"
     MYSQL_PASS="your_mysql_password"
     MYSQL_DB="at_clothes"
     ASTRA_DB_ID_MULTI_AGENT=<your_astra_db_id_multi_agent>
     ASTRA_DB_APPLICATION_TOKEN=<your_astra_db_application_token>
     ASTRA_KEYSPACE="default_keyspace"
     ASTRA_TBL="multi_agents_tbl"
     ```
7. **Run the Application**
   ```bash
   python app.py
   ```
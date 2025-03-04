from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os
import subprocess
import shutil

# Extract data from pdf
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    return loader.load()

#split text
def text_split(extract_data):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(extract_data)

#embeddings
def download_huggingface_embedding(model_name = "all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def process_pdfs(uploaded_files):
    if not uploaded_files:
        return "No files uploaded."
    
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "pdfs")
    os.makedirs(data_dir, exist_ok=True)  # Ensure the Data directory exists

    processed_files = []
    skipped_files = []

    for uploaded_file in uploaded_files:
        file_name = os.path.basename(uploaded_file)
        file_path = os.path.join(data_dir, file_name)  # Ensure safe filename
        
        # Check if file already exists
        if os.path.exists(file_path):
            print(f"Skipping {file_name}, already exists.")
            skipped_files.append(file_name)
            continue  # Skip processing the duplicate file
        
        shutil.copy(uploaded_file, file_path)  # Copy new file
        processed_files.append(file_path)
        print(f"Saved file: {file_path}")

    # Execute vectordb_store.py only if new files were processed
    if processed_files:
        subprocess.run(["python", "vectordb_store.py"], check=True)

    # Proper return statement using Python's ternary operator
    return f"File already exists." if not processed_files else f"Processed {len(processed_files)} documents. VectorDB updated."

def few_shots():
    few_shots = [
        {'Question' : "How many t-shirts do we have left for Nike in XS size and white color?",
        'SQLQuery' : "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'",
        'SQLResult': "Result of the SQL query",
        'Answer' : "91"},
        {'Question': "How much is the total price of the inventory for all S-size t-shirts?",
        'SQLQuery':"SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'",
        'SQLResult': "Result of the SQL query",
        'Answer': "22292"},
        {'Question': "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?" ,
        'SQLQuery' : """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
    (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
    group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
    """,
        'SQLResult': "Result of the SQL query",
        'Answer': "16725.4"} ,
        {'Question' : "If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?" ,
        'SQLQuery': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'",
        'SQLResult': "Result of the SQL query",
        'Answer' : "17462"},
        {'Question': "How many white color Levi's shirt I have?",
        'SQLQuery' : "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'",
        'SQLResult': "Result of the SQL query",
        'Answer' : "290"
        },
        {'Question': "how much sales amount will be generated if we sell all large size t shirts today in nike brand after discounts?",
        'SQLQuery' : """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
    (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Nike' and size="L"
    group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
    """,
        'SQLResult': "Result of the SQL query",
        'Answer' : "290"
        }
    ]
    
    return few_shots

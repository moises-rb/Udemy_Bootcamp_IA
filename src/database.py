import polars as pl
import os
from urllib.parse import quote_plus  # <--- Adicione esta importação
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    # Pegamos as variáveis do .env
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASS')
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    dbname = os.getenv('DB_NAME')

    # quote_plus transforma o '@' em '%40' para não quebrar a string de conexão
    password_encoded = quote_plus(password)
    
    db_url = f"postgresql://{user}:{password_encoded}@{host}:{port}/{dbname}"
    return db_url

def execute_query(query: str):
    uri = get_db_connection()
    # Adicionando uma configuração extra para o driver ADBC
    return pl.read_database_uri(query=query, uri=uri, engine="adbc")
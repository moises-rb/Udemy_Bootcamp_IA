import polars as pl
from src.database import execute_query

def mapeamento_visual():
    print("👀 --- MAPEAMENTO VISUAL DE COLUNAS --- 👀\n")
    
    tabelas = ["clientes", "pedidocredito", "parcelascredito", "produtosfinanciados"]
    
    for tabela in tabelas:
        print(f"📌 Tabela: {tabela}")
        # Pegamos apenas 1 linha para ver o cabeçalho e tipos
        df = execute_query(f"SELECT * FROM {tabela} LIMIT 1")
        
        # Exibe as colunas e os tipos detectados pelo Polars
        for col, dtype in zip(df.columns, df.dtypes):
            print(f"  - {col}: {dtype}")
        print("-" * 30)

if __name__ == "__main__":
    mapeamento_visual()
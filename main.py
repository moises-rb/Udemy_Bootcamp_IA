from src.database import execute_query

def run():
    print("🚀 Iniciando extração de dados...")
    
    # Exemplo: Buscar as primeiras 10 linhas de uma tabela (ajuste o nome da tabela conforme o banco)
    query = "SELECT * FROM information_schema.tables LIMIT 10" 
    
    try:
        df = execute_query(query)
        print("✅ Dados carregados com sucesso!")
        print(df.head())
        
        # Salvando uma cópia em raw para não precisar bater no banco toda hora
        df.write_parquet("data/raw/extracao_inicial.parquet")
        print("💾 Backup salvo em data/raw/")
        
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    run()
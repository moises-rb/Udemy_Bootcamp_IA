from src.database import execute_query
import polars as pl

def explorar_banco():
    print("📋 --- RECONHECIMENTO DO BANCO DE DADOS --- 📋\n")

    # 1. Listar tabelas e contagem de registros (Tamanho)
    # Nota: No Postgres, uma forma rápida de ver o 'size' é consultar as estatísticas
    query_tabelas = """
    SELECT 
        relname AS tabela, 
        n_live_tup AS total_registros 
    FROM pg_stat_user_tables 
    ORDER BY total_registros DESC
    """
    df_tabelas = execute_query(query_tabelas)
    print("1. Tabelas e Volume de Dados:")
    print(df_tabelas)
    print("-" * 30)

    # 2. Verificar Colunas, Tipos e Chaves Primárias
    # Vamos focar nas tabelas do schema 'public'
    query_colunas = """
    SELECT 
        c.table_name, 
        c.column_name, 
        c.data_type, 
        c.is_nullable,
        tc.constraint_type
    FROM information_schema.columns c
    LEFT JOIN information_schema.key_column_usage kcu 
        ON c.table_name = kcu.table_name AND c.column_name = kcu.column_name
    LEFT JOIN information_schema.table_constraints tc 
        ON kcu.constraint_name = tc.constraint_name
    WHERE c.table_schema = 'public'
    ORDER BY c.table_name
    """
    df_colunas = execute_query(query_colunas)
    print("2. Estrutura de Colunas e Chaves:")
    # Filtrando para ver as PKs (Primary Keys)
    pks = df_colunas.filter(pl.col("constraint_type") == "PRIMARY KEY")
    print(pks)
    print("-" * 30)

    # 3. Identificar Chaves Estrangeiras (Relacionamentos)
    query_fks = """
    SELECT
        tc.table_name AS tabela_origem, 
        kcu.column_name AS coluna_origem, 
        ccu.table_name AS tabela_destino,
        ccu.column_name AS coluna_destino
    FROM information_schema.table_constraints AS tc 
    JOIN information_schema.key_column_usage AS kcu
      ON tc.constraint_name = kcu.constraint_name
      AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage AS ccu
      ON ccu.constraint_name = tc.constraint_name
      AND ccu.table_schema = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY'
    """
    df_fks = execute_query(query_fks)
    print("3. Relacionamentos (Chaves Estrangeiras):")
    print(df_fks)

if __name__ == "__main__":
    explorar_banco()
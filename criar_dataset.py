import polars as pl
from src.database import execute_query
from src.processing import calcular_idade, limpar_moeda

def gerar_base_ia():
    print("🚀 Extraindo dados brutos...")
    
    # Extraímos as tabelas necessárias
    df_clientes = execute_query("SELECT * FROM clientes")
    df_pedidos = execute_query("SELECT * FROM pedidocredito WHERE status = 'Aprovado'")
    df_parcelas = execute_query("SELECT * FROM parcelascredito")
    df_produtos = execute_query("SELECT * FROM produtosfinanciados")

    print("🧠 Processando lógica de negócio com Polars...")

    # 1. Definir a Classe (Bom/Ruim) na tabela de parcelas
    # Agrupamos por solicitacaoid e verificamos se existe algum 'Vencido'
    df_target = (
        df_parcelas.group_by("solicitacaoid")
        .agg(
            pl.col("status").filter(pl.col("status") == "Vencido").count().alias("qtd_vencidos")
        )
        .with_columns(
            pl.when(pl.col("qtd_vencidos") > 0).then(pl.lit("ruim")).otherwise(pl.lit("bom")).alias("classe")
        )
    )

    # 2. Unindo as tabelas (Joins)
    df_final = (
        df_pedidos
        .join(df_clientes, on="clienteid")
        .join(df_produtos, on="produtoid")
        .join(df_target, on="solicitacaoid", how="left")
        # Se não tem parcela vencida e não está no df_target, consideramos 'bom'
        .with_columns(pl.col("classe").fill_null("bom"))
    )

    # 3. Transformações Finais (Idade e Limpeza de Moeda)
    df_final = df_final.with_columns([
        calcular_idade(pl.col("datanascimento")).alias("idade"),
        limpar_moeda(pl.col("valorsolicitado")).alias("valor_solicitado"),
        limpar_moeda(pl.col("valortotalbem")).alias("valor_total_bem")
    ])

    # 4. Selecionando apenas as colunas que o professor usou
    colunas_finais = [
        "profissao", "tempoprofissao", "renda", "tiporesidencia", 
        "escolaridade", "score", "idade", "dependentes", 
        "estadocivil", "nomecomercial", "valor_solicitado", 
        "valor_total_bem", "classe"
    ]
    
    dataset = df_final.select(colunas_finais)

    print(f"✅ Dataset criado com {dataset.shape[0]} linhas!")
    print(dataset.head())

    # Salvando em Parquet (Padrão ouro para IA)
    dataset.write_parquet("data/raw/base_treinamento.parquet")
    print("💾 Arquivo salvo em data/raw/base_treinamento.parquet")

if __name__ == "__main__":
    gerar_base_ia()
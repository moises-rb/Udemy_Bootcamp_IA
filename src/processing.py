import polars as pl
from datetime import date
from fuzzywuzzy import process
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# --- Funções de Tratamento e Pré-processamento ---
def calcular_idade(col_nascimento):
    hoje = date.today()
    return (hoje.year - col_nascimento.dt.year())

def limpar_moeda(coluna):
    return (coluna.str.replace_all(r"[^0-9,]", "").str.replace(",", ".").cast(pl.Float64))


def substituir_nulos(df: pl.DataFrame) -> pl.DataFrame:
    expressoes = []
    for coluna in df.columns:
        if df[coluna].dtype == pl.String:
            moda = df[coluna].mode()[0]
            expressoes.append(pl.col(coluna).fill_null(moda))
        else:
            mediana = df[coluna].median()
            expressoes.append(pl.col(coluna).fill_null(mediana))
    return df.with_columns(expressoes)

def corrigir_erros_digitacao(df: pl.DataFrame, coluna: str, lista_valida: list) -> pl.DataFrame:
    """
    Corrige strings em uma coluna comparando-as com uma lista de valores válidos
    usando similaridade de texto (FuzzyWuzzy).
    """
    def busca_correcao(valor):
        # Se o valor for nulo ou já estiver na lista correta, não faz nada
        if valor is None or valor in lista_valida:
            return valor
        
        # Encontra a melhor correspondência na lista_valida
        # extractOne retorna (valor, score), pegamos apenas o [0]
        correcao = process.extractOne(str(valor), lista_valida)[0]
        return correcao

    # Aplicamos a função na coluna escolhida
    return df.with_columns(
        pl.col(coluna).map_elements(busca_correcao, return_dtype=pl.String).alias(coluna)
    )

def tratar_outliers(df: pl.DataFrame, coluna: str, minimo: float, maximo: float) -> pl.DataFrame:
    mediana = df.filter((pl.col(coluna) >= minimo) & (pl.col(coluna) <= maximo))[coluna].median()
    return df.with_columns(
        pl.when((pl.col(coluna) < minimo) | (pl.col(coluna) > maximo))
        .then(mediana).otherwise(pl.col(coluna)).alias(coluna)
    )

def save_scalers(df: pl.DataFrame, colunas: list) -> pl.DataFrame:
    os.makedirs("objects", exist_ok=True)
    df_pd = df.to_pandas()
    for col in colunas:
        scaler = StandardScaler()
        df_pd[col] = scaler.fit_transform(df_pd[[col]])
        joblib.dump(scaler, f"objects/scaler_{col}.joblib")
    return pl.from_pandas(df_pd)

def save_encoders(df: pl.DataFrame, colunas: list) -> pl.DataFrame:
    os.makedirs("objects", exist_ok=True)
    df_pd = df.to_pandas()
    for col in colunas:
        le = LabelEncoder()
        df_pd[col] = le.fit_transform(df_pd[col])
        joblib.dump(le, f"objects/label_encoder_{col}.joblib")
    return pl.from_pandas(df_pd)

def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("valorsolicitado") / pl.col("valortotalbem")).alias("proporcaosolicitadototal")
    )

def load_scalers(df, nomes_colunas):
    for col in nomes_colunas:
        scaler = joblib.load(f"objects/scaler_{col}.joblib")
        df[col] = scaler.transform(df[[col]])
    return df

def load_encoders(df, nomes_colunas):
    for col in nomes_colunas:
        le = joblib.load(f"objects/label_encoder_{col}.joblib")
        df[col] = le.transform(df[col])
    return df

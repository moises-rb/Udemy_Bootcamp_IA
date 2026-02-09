import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import polars as pl
import os

# Importando suas funções de processamento
from src.processing import load_scalers, load_encoders, feature_engineering

# Título da aplicação
st.set_page_config(page_title="Análise de Crédito IA", layout="centered")
st.title("🛡️ Sistema de Previsão de Empréstimos")

# 1. Carregamento dos Modelos (Usando cache para performance)
@st.cache_resource
def load_models():
    # Carrega o modelo Keras e o seletor de atributos
    model = tf.keras.models.load_model('meu_modelo.keras')
    selector = joblib.load('objects/selector.joblib')
    return model, selector

try:
    model, selector = load_models()
except Exception as e:
    st.error(f"Erro ao carregar modelos: {e}")

# --- Definição das Opções (Igual ao seu código) ---
profissoes = ['Advogado', 'Arquiteto', 'Cientista de Dados', 'Contador', 'Dentista', 'Empresário', 'Engenheiro', 'Médico', 'Programador']
tipos_residencia = ['Alugada', 'Outros', 'Própria']
escolaridades = ['Ens.Fundamental', 'Ens.Médio', 'PósouMais', 'Superior']
scores = ['Baixo', 'Bom', 'Justo', 'MuitoBom']
estados_civis = ['Casado', 'Divorciado', 'Solteiro', 'Víuvo']
produtos = ['AgileXplorer', 'DoubleDuty', 'EcoPrestige', 'ElegantCruise', 'SpeedFury', 'TrailConqueror', 'VoyageRoamer', 'WorkMaster']

# Formuário de entrada
with st.form(key='prediction_form'):
    col_a, col_b = st.columns(2)
    
    with col_a:
        profissao = st.selectbox('Profissão', profissoes)
        tempo_profissao = st.number_input('Tempo na profissão (anos)', min_value=0, value=5)
        renda = st.number_input('Renda mensal', min_value=0.0, value=5000.0)
        tipo_residencia = st.selectbox('Tipo de residência', tipos_residencia)
        escolaridade = st.selectbox('Escolaridade', escolaridades)
        score = st.selectbox('Score de Crédito', scores)
        
    with col_b:
        idade = st.number_input('Idade', min_value=18, max_value=110, value=30)
        dependentes = st.number_input('Dependentes', min_value=0, value=0)
        estado_civil = st.selectbox('Estado Civil', estados_civis)
        produto = st.selectbox('Produto Solicitado', produtos)
        valor_solicitado = st.number_input('Valor solicitado', min_value=0.0, value=10000.0)
        valor_total_bem = st.number_input('Valor total do bem', min_value=0.0, value=20000.0)

    submit_button = st.form_submit_button(label='🚀 Analisar Crédito')

if submit_button:
    with st.spinner('Processando análise...'):
        try:
            # 2. Preparação dos Dados (Mesma lógica da sua API)
            proporcao = valor_solicitado / valor_total_bem if valor_total_bem > 0 else 0
            
            dados_dict = {
                'profissao': [profissao], 'tempoprofissao': [tempo_profissao], 'renda': [renda],
                'tiporesidencia': [tipo_residencia], 'escolaridade': [escolaridade], 'score': [score],
                'idade': [idade], 'dependentes': [dependentes], 'estadocivil': [estado_civil],
                'produto': [produto], 'valorsolicitado': [valor_solicitado], 'valortotalbem': [valor_total_bem],
                'proporcaosolicitadototal': [proporcao]
            }
            
            df = pd.DataFrame(dados_dict)

            # 3. Processamento (Scalers e Encoders)
            # Nota: Certifique-se que load_scalers busca na pasta 'objects/'
            col_numericas = ['tempoprofissao', 'renda', 'idade', 'dependentes', 
                            'valorsolicitado', 'valortotalbem', 'proporcaosolicitadototal']
            
            df = load_scalers(df, col_numericas)
            df = load_encoders(df, ['profissao', 'tiporesidencia', 'escolaridade', 'score', 'estadocivil', 'produto'])
            
            # 4. Seleção de Atributos e Predição
            df_selected = selector.transform(df)
            prediction = model.predict(df_selected)
            
            probabilidade = float(prediction[0][0])
            classe = "BOM" if probabilidade > 0.5 else "RUIM"

            # 5. Interface de Resultado
            st.markdown("---")
            c1, c2 = st.columns(2)
            c1.metric("Confiança da Aprovação", f"{probabilidade*100:.2f}%")
            
            if classe == "BOM":
                c2.success(f"### SCORE: {classe} ✅")
            else:
                c2.error(f"### SCORE: {classe} ❌")

        except Exception as e:
            st.error(f"Erro no processamento: {e}")
import polars as pl
import numpy as np
import random as python_random
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import tensorflow as tf

from src.database import execute_query
from src.const import QUERY_TREINAMENTO
from src.processing import (
    substituir_nulos,limpar_moeda, corrigir_erros_digitacao, tratar_outliers,
    feature_engineering, save_scalers, save_encoders, calcular_idade
)

# 1. Reprodutividade
seed = 41
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

# 2. Dados Brutos com Polars (Mais rápido)
df = execute_query(QUERY_TREINAMENTO)

# 3. Limpeza e Tratamento (Lógica do Professor via Polars)
profissoes_validas = ['Advogado', 'Arquiteto', 'Cientista de Dados', 'Contador','Dentista',
                      'Empresário','Engenheiro','Médico','Programador']

df = (
    # PASSO 1: Converter as Strings financeiras em Números (Float64)
    df.with_columns([
        limpar_moeda(pl.col("valorsolicitado")),
        limpar_moeda(pl.col("valortotalbem"))
    ])
    # PASSO 2: Seguir com o restante da limpeza
    .pipe(substituir_nulos)
    .pipe(corrigir_erros_digitacao, 'profissao', profissoes_validas)
    .pipe(tratar_outliers, 'tempoprofissao', 0, 70)
    .pipe(tratar_outliers, 'idade', 0, 110)
    # PASSO 3: Agora a divisão vai funcionar, pois as colunas já são f64
    .pipe(feature_engineering)
)

# 4. Divisão de Dados
# Precisamos do Pandas/Numpy para o Scikit-Learn
X = df.drop("classe").to_pandas()
y = df.select("classe").to_pandas()["classe"]

# Mapeamento manual da classe (LabelEncoder também funcionaria)
mapeamento = {'ruim': 0, 'bom': 1}
y = y.map(mapeamento)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# 5. Normalização e Codificação (Usando as funções que salvam os .joblib)
col_numericas = ['tempoprofissao','renda','idade','dependentes',
                 'valorsolicitado','valortotalbem','proporcaosolicitadototal']
col_categoricas = ['profissao', 'tiporesidencia', 'escolaridade','score','estadocivil','produto']

# Importante: transformamos e já salvamos o objeto para o futuro
X_train = save_scalers(pl.from_pandas(X_train), col_numericas).to_pandas()
X_test = save_scalers(pl.from_pandas(X_test), col_numericas).to_pandas()

X_train = save_encoders(pl.from_pandas(X_train), col_categoricas).to_pandas()
X_test = save_encoders(pl.from_pandas(X_test), col_categoricas).to_pandas()

'''
# 6. Seleção de Atributos (RFE)
print("🎯 Selecionando os melhores atributos...")
model = RandomForestClassifier(random_state=seed)
selector = RFE(model, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)

# Transforma os dados removendo as colunas menos importantes
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

# Salva o seletor
joblib.dump(selector, './objects/selector.joblib')

print("✅ Modelo e Seletores preparados!")

'''
# 7. Criação do Modelo (TensorFlow/Keras)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Configurando o otimizador
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# Compilando o modelo
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento do modelo 
model.fit(
    X_train,
    y_train,
    validation_split=0.2,  # Usa 20% dos dados para validação
    epochs=500,  # Número máximo de épocas
    batch_size=10,
    verbose=1
)
model.save('meu_modelo.keras')

# Previsões
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  

# Avaliando o modelo
print("Avaliação do Modelo nos Dados de Teste:")
model.evaluate(X_test, y_test)

# Métricas de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

import pandas as pd  # <--- Essencial para evitar o NameError

# 8. Função de Previsão simplificada para o LIME
def model_predict(data_asarray):
    # O LIME gera dados sintéticos baseados nos valores numéricos que já processamos.
    # Não precisamos re-aplicar scalers ou encoders aqui.
    predictions = model.predict(data_asarray, verbose=0)
    # Formato exigido pelo LIME: [prob_classe_0, prob_classe_1]
    return np.hstack((1 - predictions, predictions))

import lime
import lime.lime_tabular

# 9. Configuração do Explainer
# Como o RFE está comentado, X_train ainda é um DataFrame. Usamos .values para o LIME.
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, 
    feature_names=X_train.columns.tolist(), 
    class_names=['ruim', 'bom'], 
    mode='classification'
)

# 10. Gerando a explicação para o segundo registro do teste (índice 1)
print("\nExplicando a previsão com LIME...")
exp = explainer.explain_instance(X_test.values[1], model_predict, num_features=10)

# Salva o resultado em HTML para visualização no navegador
exp.save_to_file('lime_explanation.html')

# 11. Impressão dos pesos no console
print('\nRecursos e seus pesos para a classe "Bom":')
feature_importances = exp.as_list(label=1)
for feature, weight in feature_importances:
    print(f"{feature}: {weight}")
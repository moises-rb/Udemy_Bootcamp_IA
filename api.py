import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Bloqueia logs inúteis do TF

import sys
from flask import Flask, jsonify, request
import pandas as pd
import joblib
import tensorflow as tf

# 1. Ajuste do caminho para encontrar a pasta src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.processing import load_scalers, load_encoders  # Importação específica

app = Flask(__name__)

# 2. Carregar os artefatos (Certifique-se que os caminhos estão corretos)
model = tf.keras.models.load_model('meu_modelo.keras')
selector = joblib.load('objects/selector.joblib') # Adicionado caminho da pasta objects

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        
        # O segredo: Converter o dicionário de listas em um DataFrame de várias linhas
        df = pd.DataFrame(input_data)

        # 1. Aplicar a Engenharia de Atributos antes da codificação
        # (Se a sua função feature_engineering já estiver no src.processing)
        if 'proporcaosolicitadototal' not in df.columns:
             df = feature_engineering(pl.from_pandas(df)).to_pandas()

        # 2. Limpeza e Tipagem (Garantir que são floats/ints)
        col_numericas = ['tempoprofissao', 'renda', 'idade', 'dependentes', 
                        'valorsolicitado', 'valortotalbem', 'proporcaosolicitadototal']
        for col in col_numericas:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Processamento (Scalers e Encoders)
        # IMPORTANTE: Use o transform em loop para cada coluna
        df = load_scalers(df, col_numericas)
        df = load_encoders(df, ['profissao', 'tiporesidencia', 'escolaridade', 'score', 
                                'estadocivil', 'produto'])
        
        # 4. Seleção de Atributos (RFE)
        # O transform do RFE precisa que as colunas estejam na ordem EXATA do treino
        df_selected = selector.transform(df)

        # 5. Predição
        predictions = model.predict(df_selected)
        
        # 6. Formatar resposta para os 4 clientes
        respostas = []
        for p in predictions:
            prob = float(p[0])
            respostas.append({
                'probabilidade': prob,
                'classe': 'bom' if prob > 0.5 else 'ruim'
            })
        
        return jsonify({'status': 'sucesso', 'resultados': respostas})

    except Exception as e:
        return jsonify({'erro': str(e)}), 400
    
if __name__ == '__main__':
    # O uso do debug=False às vezes ajuda a estabilizar o carregamento do TensorFlow no Windows
    app.run(host='0.0.0.0', port=5000, debug=False)
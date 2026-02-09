# 🛡️ Sistema Inteligente de Aprovação de Crédito

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

Este projeto apresenta uma solução **End-to-End** para análise de risco de crédito, utilizando técnicas avançadas de Machine Learning e Deep Learning.  
A aplicação vai desde o processamento de grandes volumes de dados até a entrega de uma interface visual interativa para tomada de decisão.

---

## 🚀 Funcionalidades

- **Pipeline de Dados de Alta Performance:** Utilização da biblioteca `Polars` para manipulação eficiente de dados.  
- **Modelo de Deep Learning:** Rede Neural desenvolvida em `TensorFlow/Keras` para classificação de risco (Bom/Mau pagador).  
- **Feature Selection:** Implementação de `RFE` (Recursive Feature Elimination) para identificar as variáveis mais relevantes para o negócio.  
- **Explicabilidade (XAI):** Integração com `LIME` (Local Interpretable Model-agnostic Explanations) para justificar as decisões da IA.  
- **Interface Web:** Dashboard interativo construído em `Streamlit` para consultas em tempo real.  

---

## 🛠️ Arquitetura do Projeto

```text
📂 Udemy_Bootcamp_IA/
├── 📂 objects/             # Scalers, Encoders e Seletores (.joblib)
├── 📂 src/                 # Funções modulares de processamento
│   └── processing.py
├── meu_modelo.keras        # Modelo de rede neural treinado
├── webapp.py               # Interface Streamlit
├── api.py                  # API Flask para integração (opcional)
├── requirements.txt        # Dependências do projeto
└── README.md
```

---

## 🔧 Como Executar

### 1. Clonar o repositório
```bash
git clone https://github.com/moises-rb/Udemy_Bootcamp_IA.git
cd Udemy_Bootcamp_IA
```

### 2. Configurar o ambiente virtual
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate    # Windows
```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
```

### 4. Rodar a aplicação
```bash
streamlit run webapp.py
```

---

## 🧠 Metodologia Técnica

- **Engenharia de Atributos:**  
  Criação de métricas financeiras como a `proporcaosolicitadototal`.

- **Pré-processamento:**  
  Normalização de dados numéricos e codificação de variáveis categóricas preservando a integridade dos dados de treino/teste.

- **Treinamento:**  
  Otimização de hiperparâmetros para garantir alta acurácia e baixo índice de falsos positivos em concessão de crédito.

---

## 👨‍💻 Autor

Desenvolvido por **Moisés Ribeiro** durante o Bootcamp de IA Aplicada.  

🔗 LinkedIn: https://www.linkedin.com/in/moisesrsjr/

---

⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!

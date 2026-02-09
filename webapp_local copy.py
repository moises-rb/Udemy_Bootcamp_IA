import streamlit as st
import requests

# URL padrão do Flask rodando localmente
url = "http://localhost:5000/predict"

st.title("Cliente para API de Previsão de Empréstimos")

profissoes = ['Advogado', 'Arquiteto', 'Cientista de Dados', 
              'Contador', 'Dentista', 'Empresário', 'Engenheiro', 'Médico', 'Programador']
tipos_residencia = ['Alugada', 'Outros', 'Própria']
escolaridades = ['Ens.Fundamental', 'Ens.Médio', 'PósouMais', 'Superior']
scores = ['Baixo', 'Bom', 'Justo', 'MuitoBom']
estados_civis = ['Casado', 'Divorciado', 'Solteiro', 'Víuvo']
produtos = ['AgileXplorer', 'DoubleDuty', 'EcoPrestige', 'ElegantCruise', 
            'SpeedFury', 'TrailConqueror', 'VoyageRoamer', 'WorkMaster']

with st.form(key='prediction_form'):
    profissao = st.selectbox('Profissão', profissoes)
    tempo_profissao = st.number_input('Tempo na profissão (em anos)', min_value=0, value=0, step=1)
    renda = st.number_input('Renda mensal', min_value=0.0, value=0.0, step=1000.0)
    tipo_residencia = st.selectbox('Tipo de residência', tipos_residencia)
    escolaridade = st.selectbox('Escolaridade', escolaridades)
    score = st.selectbox('Score', scores)
    idade = st.number_input('Idade', min_value=18, max_value=110, value=25, step=1)
    dependentes = st.number_input('Dependentes', min_value=0, value=0, step=1)
    estado_civil = st.selectbox('Estado Civil', estados_civis)
    produto = st.selectbox('Produto', produtos)
    valor_solicitado = st.number_input('Valor solicitado', min_value=0.0, 
                                       value=0.0, step=1000.0)
    valor_total_bem = st.number_input('Valor total do bem', min_value=0.0, 
                                      value=0.0, step=1000.0)

    submit_button = st.form_submit_button(label='Consultar')

if submit_button:
    # Cálculo preventivo para evitar divisão por zero
    proporcao = valor_solicitado / valor_total_bem if valor_total_bem > 0 else 0

    dados_novos = {
        'profissao': [profissao],
        'tempoprofissao': [tempo_profissao],
        'renda': [renda],
        'tiporesidencia': [tipo_residencia],
        'escolaridade': [escolaridade],
        'score': [score],
        'idade': [idade],
        'dependentes': [dependentes],
        'estadocivil': [estado_civil],
        'produto': [produto],
        'valorsolicitado': [valor_solicitado],
        'valortotalbem': [valor_total_bem],
        'proporcaosolicitadototal': [proporcao]
    }

    with st.spinner('Analisando perfil de crédito...'):
        try:
            response = requests.post(url, json=dados_novos)
            
            if response.status_code == 200:
                # Ajuste conforme a estrutura do JSON que vimos no teste_flask
                res = response.json()
                dados_pred = res['resultados'][0] # Pega o primeiro (e único) resultado
                
                probabilidade = dados_pred['probabilidade'] * 100
                classe = dados_pred['classe'].upper()

                # Visual impactante
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Confiança da IA", value=f"{probabilidade:.2f}%")
                
                with col2:
                    if classe == "BOM":
                        st.success(f"### RESULTADO: {classe} ✅")
                    else:
                        st.error(f"### RESULTADO: {classe} ❌")
                
                # Opcional: Mostrar uma dica baseada na probabilidade
                if probabilidade > 90:
                    st.info("💡 Este cliente apresenta um perfil de baixíssimo risco.")
                    
            else:
                st.error(f"Erro na API: {response.status_code} - {response.text}")
        
        except Exception as e:
            st.error(f"Falha na conexão com o servidor: {e}")
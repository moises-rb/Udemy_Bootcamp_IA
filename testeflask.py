import requests
from environs import Env

# Inicializa o ambiente
env = Env()
env.read_env() # Ele procura automaticamente o arquivo .env na raiz

# Busca a URL (verifique se a chave no seu .env é exatamente esta)
url = env.str("url_api.url", "http://localhost:5000/predict") 

dados_novos = {
    "profissao": ["Advogado","Médico","Dentista","Contador"],
    "tempoprofissao": [39, 37, 16, 0],
    "renda": [20860.0, 5000, 20000, 7000],
    "tiporesidencia": ["Alugada","Própria","Própria","Alugada"],
    "escolaridade": ["Ens.Fundamental","PósouMais","Superior","Ens.Fundamental"],
    "score": ["Baixo","Baixo","MuitoBom","MuitoBom"],
    "idade": [36, 25, 19, 24],
    "dependentes": [0, 0, 4, 2],
    "estadocivil": ["Víuvo","Casado","Casado","Solteiro"],
    "produto": ["DoubleDuty","SpeedFury","ElegantCruise","TrailConqueror"],
    "valorsolicitado": [139244.0, 100000, 50000, 200000],
    "valortotalbem": [320000.0, 200000, 200000, 300000],
    "proporcaosolicitadototal": [2.2, 50, 200, 40]
}

try:
    response = requests.post(url, json=dados_novos)

    if response.status_code == 200:
        print("✅ Previsões recebidas:")
        predictions = response.json()
        print(predictions)
    else:
        print(f"❌ Erro na API (Status {response.status_code}):")
        print(response.text)
except Exception as e:
    print(f"🔥 Erro ao conectar na API: {e}")
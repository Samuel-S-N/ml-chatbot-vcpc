# Exercício 1: Classificador de Intenções Baseado em TF-IDF
# ORIENTAÇÃO: Loop de chat que classifica a intenção com TF-IDF + Naive Bayes.
# OBJETIVO: Substituir if/else por pipeline de PLN que infere o objetivo em tempo real.
# TESTE: Digite 3 mensagens e colete as respostas

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

PASTA = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(PASTA, "logs_ecommerce.csv")
if not os.path.exists(CSV):
    CSV = os.path.join(os.path.dirname(PASTA), "logs_ecommerce.csv")

RESPOSTAS = {
    "saudacao": (
        "Olá! Sou o assistente da loja. Posso ajudar com pedidos, rastreio "
        "ou reclamações. Como posso ajudar?"
    ),
    "suporte": (
        "Entendi que você precisa de suporte com pedido ou entrega. "
        "Informe o número do pedido ou acesse 'Meus Pedidos' no site."
    ),
    "reclamacao": (
        "Lamento pelo transtorno. Vou encaminhar sua reclamação ao time "
        "de atendimento. Pode descrever o que aconteceu com mais detalhes?"
    ),
}


def rotular_basico(msg):
    if "Oi" in msg or "Olá" in msg or "oi" in msg.lower():
        return "saudacao"
    if "pedido" in msg.lower() or "rastreio" in msg.lower() or "endereço" in msg.lower():
        return "suporte"
    return "reclamacao"


def treinar_classificador():
  df = pd.read_csv(CSV)
  df["intencao"] = df["mensagem_usuario"].apply(rotular_basico)

  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(df["mensagem_usuario"])
  y = df["intencao"]

  modelo = MultinomialNB()
  modelo.fit(X, y)
  return vectorizer, modelo


def classificar_intencao(mensagem, vectorizer, modelo):
  X_teste = vectorizer.transform([mensagem])
  return modelo.predict(X_teste)[0]


def iniciar_chatbot():
  print("=" * 55)
  print("  ASSISTENTE E-COMMERCE — Exercício 01")
  print("  Classificador TF-IDF + Naive Bayes")
  print("=" * 55)
  print("  Digite sua mensagem. Use 'sair' para encerrar.\n")

  vectorizer, modelo = treinar_classificador()
  turno = 0

  while True:
    try:
      entrada = input("Você: ").strip()
    except (EOFError, KeyboardInterrupt):
      print("\n[Sistema encerrado]")
      break

    if not entrada:
      print("Bot: Por favor, digite alguma mensagem.\n")
      continue

    if entrada.lower() == "sair":
      print(f"Bot: Até logo! Foram {turno} mensagens classificadas.")
      break

    turno += 1
    intencao = classificar_intencao(entrada, vectorizer, modelo)
    resposta = RESPOSTAS.get(intencao, RESPOSTAS["reclamacao"])

    print(f"Bot: Identifiquei que sua intenção é [{intencao}].")
    print(f"Bot: {resposta}\n")


if __name__ == "__main__":
  iniciar_chatbot()

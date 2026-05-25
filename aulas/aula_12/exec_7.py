# Exercício 7: RAG Primitivo (Retrieval-Augmented Generation) com Consulta a Arquivo CSV
# ORIENTAÇÃO: Crie um bot que pede o ID de usuário no terminal, busca as informações financeiras
#   desse ID no arquivo logs_ecommerce.csv e usa esses dados na resposta.
# OBJETIVO: Entender a base de um sistema RAG: consultar fontes de dados frias para injetar
#   contexto na resposta oferecida ao usuário final.
# TESTE: Digite 3 mensagens e colete as respostas

import os

import pandas as pd

PASTA = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(PASTA, "logs_ecommerce.csv")
if not os.path.exists(CSV):
    CSV = os.path.join(os.path.dirname(PASTA), "logs_ecommerce.csv")


class ChatbotRAG:
    """
    RAG primitivo: recupera dados do CSV por id_usuario e injeta no prompt de resposta.
    """

    def __init__(self, caminho_csv: str):
        self.df = pd.read_csv(caminho_csv)
        self.id_ativo: int | None = None
        self.contexto: dict | None = None

    def buscar_usuario(self, id_usuario: int) -> dict | None:
        registro = self.df[self.df["id_usuario"] == id_usuario]

        if registro.empty:
            return None

        ultimo = registro.iloc[-1]
        return {
            "id_usuario": int(id_usuario),
            "ultima_compra": float(ultimo["historico_compras_valor"]),
            "gasto_medio": round(registro["historico_compras_valor"].mean(), 2),
            "score_satisfacao": round(registro["score_satisfacao"].mean(), 1),
            "categoria_favorita": ultimo["categoria_produto"],
            "ultima_mensagem": ultimo["mensagem_usuario"],
            "total_registros": len(registro),
        }

    def autenticar(self, entrada: str) -> str:
        if not entrada.isdigit():
            return "Por favor, insira apenas números (ID de 4 dígitos)."

        id_usuario = int(entrada)
        contexto = self.buscar_usuario(id_usuario)

        if contexto is None:
            return f"Não localizei o ID {id_usuario} na nossa base de logs. Tente outro."

        self.id_ativo = id_usuario
        self.contexto = contexto
        return (
            f"Dados recuperados do CSV para o ID {id_usuario}!\n"
            f"  • Última compra: R$ {contexto['ultima_compra']:.2f}\n"
            f"  • Gasto médio no histórico: R$ {contexto['gasto_medio']:.2f}\n"
            f"  • Satisfação média: {contexto['score_satisfacao']}/5\n"
            f"  • Categoria mais recente: {contexto['categoria_favorita']}\n"
            f"Agora pode perguntar sobre seu histórico, gastos ou satisfação."
        )

    def responder_com_contexto(self, mensagem: str) -> str:
        if self.contexto is None:
            return "Informe primeiro seu ID de usuário de 4 dígitos."

        ctx = self.contexto
        texto = mensagem.lower()

        if any(p in texto for p in ("gasto", "compra", "valor", "quanto", "financeiro")):
            return (
                f"[Contexto CSV — ID {ctx['id_usuario']}] "
                f"Sua última compra foi de R$ {ctx['ultima_compra']:.2f} e o gasto médio "
                f"registrado é R$ {ctx['gasto_medio']:.2f} em {ctx['total_registros']} interações."
            )

        if any(p in texto for p in ("satisfação", "satisfacao", "nota", "avaliação", "avaliacao")):
            return (
                f"[Contexto CSV — ID {ctx['id_usuario']}] "
                f"Seu nível de satisfação histórico é {ctx['score_satisfacao']}/5."
            )

        if any(p in texto for p in ("categoria", "produto", "loja")):
            return (
                f"[Contexto CSV — ID {ctx['id_usuario']}] "
                f"Sua categoria mais recente nos logs é {ctx['categoria_favorita']}."
            )

        if any(p in texto for p in ("última", "ultima", "mensagem", "pedido", "histórico", "historico")):
            return (
                f"[Contexto CSV — ID {ctx['id_usuario']}] "
                f"Última mensagem registrada: \"{ctx['ultima_mensagem']}\"."
            )

        return (
            f"[Contexto CSV — ID {ctx['id_usuario']}] "
            f"Resumo: última compra R$ {ctx['ultima_compra']:.2f}, "
            f"satisfação {ctx['score_satisfacao']}/5, categoria {ctx['categoria_favorita']}. "
            f"Pergunte sobre gastos, satisfação ou histórico."
        )


def iniciar_chatbot():
    print("=" * 55)
    print("  ASSISTENTE E-COMMERCE — Exercício 07")
    print("  RAG primitivo com consulta a logs_ecommerce.csv")
    print("=" * 55)
    print(f"  Base de dados: {CSV}")
    print("  Digite 'sair' para encerrar.\n")

    bot = ChatbotRAG(CSV)
    print("Bot: Por favor, informe seu ID de usuário de 4 dígitos para consulta:")
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
            print(f"Bot: Até logo! Foram {turno} mensagens processadas.")
            break

        turno += 1

        if bot.contexto is None:
            resposta = bot.autenticar(entrada)
        else:
            resposta = bot.responder_com_contexto(entrada)

        print(f"Bot: {resposta}\n")


if __name__ == "__main__":
    iniciar_chatbot()

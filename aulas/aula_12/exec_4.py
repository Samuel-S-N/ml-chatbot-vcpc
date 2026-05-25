# Exercício 4: Análise de Sentimento com Gatilho para Transbordo Humano
# ORIENTAÇÃO: Desenvolva um chat contínuo que avalia a carga emocional das palavras digitadas
#   pelo usuário e encerra o bot se detectar hostilidade.
# OBJETIVO: Criar regras de segurança baseadas em pontuação léxica de sentimentos para transferir
#   o atendimento para um humano caso o cliente demonstre irritação severa.
# TESTE: Digite 3 mensagens e colete as respostas

LIMIAR_TRANSBORDO = 2
SINAL_TRANSBORDO = "TRANSBORDO"

PALAVRAS_NEGATIVAS = [
    "péssimo", "quebrado", "atrasou", "horroroso", "ruim", "ódio", "droga",
    "absurdo", "inaceitável", "inaceitavel", "vergonha", "raiva", "odeio",
]


def calcular_irritacao(mensagem: str) -> int:
    texto = mensagem.lower()
    return sum(1 for palavra in PALAVRAS_NEGATIVAS if palavra in texto)


class ChatbotSentimento:
    """Avalia irritação léxica e dispara transbordo humano acima do limiar."""

    def __init__(self, limiar: int = LIMIAR_TRANSBORDO):
        self.limiar = limiar
        self.transbordo_ativado = False

    def responder(self, mensagem: str) -> str:
        mensagem = mensagem.strip()
        if not mensagem:
            return "Por favor, descreva como posso ajudar."

        pontuacao = calcular_irritacao(mensagem)

        if pontuacao >= self.limiar:
            self.transbordo_ativado = True
            return (
                f"Detectei insatisfação severa (índice de irritação: {pontuacao}). "
                "Protocolo de transbordo ativado. Transferindo para supervisor humano. "
                "Aguarde — um atendente assumirá em instantes."
            )

        if pontuacao == 1:
            return (
                f"Entendi sua frustração (índice: {pontuacao}). "
                "Vou processar sua solicitação com prioridade. "
                "Se precisar de mais ajuda, estou aqui."
            )

        return "Certo, entendi. Processando sua solicitação normalmente."


def iniciar_chatbot():
    print("=" * 55)
    print("  ASSISTENTE E-COMMERCE — Exercício 04")
    print("  Análise de sentimento + transbordo humano")
    print("=" * 55)
    print(f"  Irritação ≥ {LIMIAR_TRANSBORDO} palavras negativas → atendente humano.")
    print("  Digite 'sair' para encerrar.\n")

    bot = ChatbotSentimento()
    print("Bot: Como posso ajudar você hoje?")
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
            print(f"Bot: Até logo! Foram {turno} mensagens analisadas.")
            break

        turno += 1
        pontuacao = calcular_irritacao(entrada)
        resposta = bot.responder(entrada)

        print(f"Bot: {resposta}")
        print(f"     [índice de irritação: {pontuacao}]\n")

        if bot.transbordo_ativado:
            print("[Sistema] Sessão encerrada — fila de atendimento humano.")
            break


if __name__ == "__main__":
    iniciar_chatbot()

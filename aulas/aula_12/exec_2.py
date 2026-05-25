# Exercício 2: Gerenciamento de Contexto e Memória de Curto Prazo
# ORIENTAÇÃO: Construa um fluxo de chat via prompt que pergunte primeiro o nome do usuário
#   e use essa informação nas respostas seguintes, identificando também intenções de compra.
# OBJETIVO: Compreender como arquitetar o gerenciamento de estados (Session State) para que
#   o bot não sofra de "amnésia" a cada nova linha de comando do terminal.
# TESTE: Digite 3 mensagens e colete as respostas


class ChatbotComMemoria:
    """Mantém contexto da sessão (nome e último assunto) entre turnos do terminal."""

    PALAVRAS_COMPRA = ("comprar", "preço", "preco", "promoção", "promocao", "oferta", "desconto")

    def __init__(self):
        self.contexto = {"nome": None, "ultimo_assunto": None}

    def _detectar_intencao_compra(self, mensagem: str) -> bool:
        texto = mensagem.lower()
        return any(palavra in texto for palavra in self.PALAVRAS_COMPRA)

    def responder(self, mensagem: str) -> str:
        mensagem = mensagem.strip()
        if not mensagem:
            return "Por favor, digite alguma mensagem."

        # Primeiro turno: registra o nome e personaliza a saudação
        if not self.contexto["nome"]:
            self.contexto["nome"] = mensagem
            return f"Prazer, {mensagem}! Em que posso te ajudar hoje?"

        if self._detectar_intencao_compra(mensagem):
            self.contexto["ultimo_assunto"] = "vendas"
            return (
                f"Olha, {self.contexto['nome']}, nosso setor de "
                f"{self.contexto['ultimo_assunto']} está com promoções hoje!"
            )

        self.contexto["ultimo_assunto"] = mensagem
        return f"Entendi sua mensagem sobre '{mensagem}', {self.contexto['nome']}."


def iniciar_chatbot():
    print("=" * 55)
    print("  ASSISTENTE E-COMMERCE — Exercício 02")
    print("  Memória de curto prazo (Session State)")
    print("=" * 55)
    print("  O bot lembra seu nome e o último assunto nesta sessão.")
    print("  Digite 'sair' para encerrar.\n")

    bot = ChatbotComMemoria()
    print("Bot: Olá! Qual é o seu nome?")
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
            nome = bot.contexto["nome"] or "visitante"
            print(f"Bot: Até logo, {nome}! Foram {turno} mensagens nesta sessão.")
            break

        turno += 1
        resposta = bot.responder(entrada)
        print(f"Bot: {resposta}\n")


if __name__ == "__main__":
    iniciar_chatbot()

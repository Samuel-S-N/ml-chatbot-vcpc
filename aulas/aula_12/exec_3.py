# Exercício 3: Extração de Entidades via Regex no Fluxo de Atendimento
# ORIENTAÇÃO: Implemente um bot de triagem que aguarda o usuário digitar uma mensagem
#   contendo um número de protocolo ou pedido iniciado por '#' seguido de 4 números.
# OBJETIVO: Usar Expressões Regulares (re) integradas ao input do terminal para capturar
#   e isolar dados estruturados inseridos em textos livres.
# TESTE: Digite 3 mensagens e colete as respostas

import re

# Padrão: # seguido de exatamente 4 dígitos (ex.: #1234)
PADRAO_PEDIDO = re.compile(r"#\d{4}")

# Simulação de base de pedidos (status + detalhe para o usuário)
STATUS_SISTEMA = {
    "#1234": ("em separação", "aguardando conferência no centro de distribuição"),
    "#4521": ("em transporte", "saiu do hub regional — previsão de entrega em 2 dias úteis"),
    "#5678": ("despachado", "postado nos Correios — código de rastreio enviado por e-mail"),
    "#9999": ("entregue", "recebido no endereço cadastrado em 12/05/2026"),
    "#8888": ("aguardando pagamento", "boleto ou PIX ainda não confirmado"),
}

STATUS_PADRAO = [
    ("em separação", "pedido confirmado e em preparação"),
    ("despachado", "enviado à transportadora"),
    ("em transporte", "a caminho do seu endereço"),
    ("entregue", "entrega concluída"),
]


class ChatbotTriagem:
    """Extrai números de pedido (#NNNN) de mensagens em linguagem natural."""

    def extrair_pedidos(self, mensagem: str) -> list[str]:
        return PADRAO_PEDIDO.findall(mensagem)

    def consultar_status(self, codigo: str) -> tuple[str, str]:
        if codigo in STATUS_SISTEMA:
            return STATUS_SISTEMA[codigo]
        # Fallback determinístico para qualquer #NNNN não cadastrado
        idx = int(codigo[1:]) % len(STATUS_PADRAO)
        return STATUS_PADRAO[idx]

    def _formatar_pedido(self, codigo: str) -> str:
        status, detalhe = self.consultar_status(codigo)
        return f"{codigo} — status: {status} ({detalhe})"

    def responder(self, mensagem: str) -> str:
        mensagem = mensagem.strip()
        if not mensagem:
            return "Por favor, descreva o problema e informe o número do pedido."

        pedidos = self.extrair_pedidos(mensagem)

        if pedidos:
            principal = pedidos[0]
            status, detalhe = self.consultar_status(principal)
            resposta = (
                f"Sucesso! Encontrei o pedido {principal} na sua mensagem. "
                f"Status atual: {status}. {detalhe.capitalize()}."
            )
            if len(pedidos) > 1:
                outros = [self._formatar_pedido(c) for c in pedidos[1:]]
                resposta += " Outros pedidos na mensagem: " + "; ".join(outros) + "."
            return resposta

        return (
            "Não consegui identificar o número do seu pedido. "
            "Lembre-se de usar o formato #1234 (hash + 4 números)."
        )


def iniciar_chatbot():
    print("=" * 55)
    print("  ASSISTENTE E-COMMERCE — Exercício 03")
    print("  Extração de entidades com Regex")
    print("=" * 55)
    print("  Informe o problema e o número do pedido (Ex: #1234).")
    print("  Digite 'sair' para encerrar.\n")

    bot = ChatbotTriagem()
    print("Bot: Olá! Por favor, informe o problema e o número do seu pedido (Ex: #1234).")
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
        resposta = bot.responder(entrada)
        print(f"Bot: {resposta}\n")


if __name__ == "__main__":
    iniciar_chatbot()

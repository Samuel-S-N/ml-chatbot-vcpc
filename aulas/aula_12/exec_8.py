# Exercício 8: Bot Orquestrador de Chamadas de Funções do Sistema (Router)
# ORIENTAÇÃO: Crie um bot que lê a mensagem digitada pelo usuário e escolhe automaticamente
#   se deve acionar uma função interna do sistema ou responder como texto padrão.
# OBJETIVO: Implementar a lógica primária de Roteamento de Intenções para "Tool Use"
#   (uso de ferramentas externas por bots).
# TESTE: Digite 3 mensagens e colete as respostas


def acionar_modulo_financeiro() -> str:
    return "[API BANCO] Conectando ao gateway de pagamento... Nenhum estorno pendente."


def acionar_modulo_logistica() -> str:
    return "[API LOGÍSTICA] Consultando transportadora... Pedido em trânsito, previsão 2 dias úteis."


def acionar_modulo_suporte() -> str:
    return "[API CRM] Abrindo ticket de atendimento... Protocolo #8842 criado com prioridade normal."


def acionar_modulo_catalogo() -> str:
    return "[API CATÁLOGO] Buscando produtos... 12 itens em promoção na categoria informada."


FERRAMENTAS = [
    {
        "nome": "financeiro",
        "keywords": ("dinheiro", "estorno", "pagamento", "boleto", "pix", "reembolso", "cartão", "cartao"),
        "funcao": acionar_modulo_financeiro,
    },
    {
        "nome": "logistica",
        "keywords": ("pedido", "rastreio", "entrega", "encomenda", "transporte", "despacho"),
        "funcao": acionar_modulo_logistica,
    },
    {
        "nome": "suporte",
        "keywords": ("reclamação", "reclamacao", "problema", "defeito", "quebrado", "atendente", "ajuda"),
        "funcao": acionar_modulo_suporte,
    },
    {
        "nome": "catalogo",
        "keywords": ("produto", "comprar", "promoção", "promocao", "oferta", "estoque", "preço", "preco"),
        "funcao": acionar_modulo_catalogo,
    },
]


class ChatbotOrquestrador:
    """
    Router de intenções: direciona a mensagem para a ferramenta interna correta
    ou responde pelo canal de atendimento comum.
    """

    def __init__(self, ferramentas: list[dict] = None):
        self.ferramentas = ferramentas or FERRAMENTAS
        self.ultima_ferramenta: str | None = None

    def detectar_ferramenta(self, mensagem: str) -> dict | None:
        texto = mensagem.lower()
        for ferramenta in self.ferramentas:
            if any(palavra in texto for palavra in ferramenta["keywords"]):
                return ferramenta
        return None

    def rotear(self, mensagem: str) -> tuple[str, str]:
        """
        Retorna (modo, resposta).
        modo: 'tool' ou 'texto'
        """
        mensagem = mensagem.strip()
        if not mensagem:
            return "texto", "Por favor, descreva como posso ajudar."

        ferramenta = self.detectar_ferramenta(mensagem)

        if ferramenta:
            self.ultima_ferramenta = ferramenta["nome"]
            retorno_api = ferramenta["funcao"]()
            resposta = (
                f"Detectei demanda de [{ferramenta['nome'].upper()}]. "
                f"Acionando ferramenta → {retorno_api}"
            )
            return "tool", resposta

        self.ultima_ferramenta = None
        return (
            "texto",
            "Entendido. Tratando requisição no canal de atendimento comum. "
            "Posso ajudar com pedidos, pagamentos, produtos ou suporte — "
            "basta mencionar o assunto.",
        )


def iniciar_chatbot():
    print("=" * 55)
    print("  ASSISTENTE E-COMMERCE — Exercício 08")
    print("  Orquestrador / Router de Ferramentas (Tool Use)")
    print("=" * 55)
    print("  O bot roteia automaticamente para módulos internos:")
    print("  financeiro | logística | suporte | catálogo")
    print("  Digite 'sair' para encerrar.\n")

    bot = ChatbotOrquestrador()
    print("Bot: Como posso te ajudar? (Tente falar sobre 'dinheiro', 'rastreio' ou 'produto')")
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
            print(f"Bot: Até logo! Foram {turno} mensagens roteadas.")
            break

        turno += 1
        modo, resposta = bot.rotear(entrada)

        print(f"Bot: {resposta}")
        ferramenta = bot.ultima_ferramenta or "nenhuma"
        print(f"     [modo: {modo} | ferramenta: {ferramenta}]\n")


if __name__ == "__main__":
    iniciar_chatbot()

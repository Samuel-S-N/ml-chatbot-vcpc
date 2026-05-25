# Exercício 6: FAQ Inteligente Usando Similaridade de Cosseno via Terminal
# ORIENTAÇÃO: Permita que o usuário faça perguntas livres no prompt e compare a string
#   digitada com uma base estática de dúvidas frequentes (FAQ).
# OBJETIVO: Aplicar álgebra linear e vetorização de palavras para mapear a proximidade
#   geométrica da dúvida e retornar a resposta correta mesmo se as palavras exatas mudarem.
# TESTE: Digite 3 mensagens e colete as respostas

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD = 0.25

FAQ_BASE = [
    {
        "pergunta": "Como posso rastrear meu pedido",
        "resposta": (
            "Acesse 'Meus Pedidos' no site ou no app e clique em Rastrear. "
            "O código de rastreio também é enviado por e-mail após o despacho."
        ),
    },
    {
        "pergunta": "onde está minha encomenda código rastreio",
        "resposta": (
            "Acesse 'Meus Pedidos' no site ou no app e clique em Rastrear. "
            "O código de rastreio também é enviado por e-mail após o despacho."
        ),
    },
    {
        "pergunta": "Quais as formas de pagamento aceitas",
        "resposta": (
            "Aceitamos cartão de crédito e débito, PIX, boleto bancário e "
            "cartões de lojas parceiras. Parcelamento em até 12x no crédito."
        ),
    },
    {
        "pergunta": "aceitam pix cartão boleto pagamento",
        "resposta": (
            "Aceitamos cartão de crédito e débito, PIX, boleto bancário e "
            "cartões de lojas parceiras. Parcelamento em até 12x no crédito."
        ),
    },
    {
        "pergunta": "Como funciona a política de troca",
        "resposta": (
            "Você tem até 30 dias após o recebimento para solicitar troca ou devolução. "
            "O produto deve estar na embalagem original e sem uso. "
            "Abra uma solicitação em 'Minha Conta' > 'Trocas e Devoluções'."
        ),
    },
    {
        "pergunta": "quero devolver trocar produto compra",
        "resposta": (
            "Você tem até 30 dias após o recebimento para solicitar troca ou devolução. "
            "O produto deve estar na embalagem original e sem uso. "
            "Abra uma solicitação em 'Minha Conta' > 'Trocas e Devoluções'."
        ),
    },
    {
        "pergunta": "Qual o prazo de entrega",
        "resposta": (
            "O prazo varia por região: capitais de 3 a 7 dias úteis; "
            "interior de 7 a 15 dias úteis. O prazo exato aparece no checkout."
        ),
    },
    {
        "pergunta": "Como cancelar um pedido",
        "resposta": (
            "Pedidos ainda não despachados podem ser cancelados em 'Meus Pedidos'. "
            "Após o envio, aguarde o recebimento e solicite devolução."
        ),
    },
]


class ChatbotFAQ:
    """
    FAQ com CountVectorizer + similaridade do cosseno.
    Encontra a pergunta mais próxima mesmo com palavras diferentes.
    """

    def __init__(self, faq: list[dict], threshold: float = THRESHOLD):
        self.faq = faq
        self.threshold = threshold
        self.perguntas = [item["pergunta"].lower() for item in faq]

        self.vectorizer = CountVectorizer(ngram_range=(1, 2))
        self.matriz_faq = self.vectorizer.fit_transform(self.perguntas)

    def buscar_similaridade(self, entrada: str) -> tuple[int, float]:
        """
        Compara a entrada com o FAQ via cosseno.
        Equivalente ao TODO: cosine_similarity(vetores[-1:], vetores[:-1])
        após vetorizar FAQ + entrada no mesmo espaço.
        """
        todas_frases = self.perguntas + [entrada.lower()]
        matriz = self.vectorizer.fit_transform(todas_frases).toarray()
        similitudes = cosine_similarity(matriz[-1:], matriz[:-1])
        melhor_indice = int(similitudes.argmax())
        melhor_score = float(similitudes[0][melhor_indice])
        return melhor_indice, melhor_score

    def responder(self, mensagem: str) -> tuple[str, float, str]:
        mensagem = mensagem.strip()
        if not mensagem:
            return "Por favor, digite sua dúvida.", 0.0, "N/A"

        idx, score = self.buscar_similaridade(mensagem)
        pergunta_ref = self.faq[idx]["pergunta"]

        if score >= self.threshold:
            return self.faq[idx]["resposta"], round(score, 4), pergunta_ref

        return (
            "Desculpe, não localizei uma resposta exata no FAQ. "
            "Tente reformular ou pergunte sobre rastreio, pagamento ou trocas.",
            round(score, 4),
            pergunta_ref,
        )


def iniciar_chatbot():
    print("=" * 55)
    print("  ASSISTENTE E-COMMERCE — Exercício 06")
    print("  FAQ com CountVectorizer + Similaridade de Cosseno")
    print("=" * 55)
    print(f"  Confiança mínima: {THRESHOLD}. Digite 'sair' para encerrar.\n")

    bot = ChatbotFAQ(FAQ_BASE)
    print(f"  ✓ FAQ carregado: {len(FAQ_BASE)} entradas na base\n")
    print("Bot: Digite sua dúvida sobre nossa operação (rastreio, pagamento, troca...):")
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
            print(f"Bot: Até logo! Foram {turno} dúvidas processadas.")
            break

        turno += 1
        resposta, score, pergunta_ref = bot.responder(entrada)

        print(f"Bot: {resposta}")
        print(f"     [similaridade: {score:.4f} | FAQ mais próximo: '{pergunta_ref}']\n")


if __name__ == "__main__":
    iniciar_chatbot()

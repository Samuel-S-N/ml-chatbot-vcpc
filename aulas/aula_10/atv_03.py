# ============================================================
# EXERCÍCIO 3 — Chatbot Retrieval-Based
# Modelo: FAQ com similaridade de strings (difflib)
# ============================================================
from difflib import SequenceMatcher

# Base de conhecimento: perguntas e respostas
FAQ = {
    'como faço para criar uma conta':   'Acesse o site, clique em Cadastre-se e preencha o formulário.',
    'qual o prazo de entrega':           'O prazo padrão é de 5 a 10 dias úteis.',
    'como rastrear meu pedido':          'Acesse Minha Conta > Pedidos > Rastrear.',
    'posso trocar um produto':            'Sim! Trocas são aceitas em até 30 dias com nota fiscal.',
    'como cancelar minha compra':        'Pedidos podem ser cancelados em até 24h após a compra.',
    'quais formas de pagamento':         'Aceitamos cartão, Pix e boleto bancário.',
    'tem frete gratis':                  'Frete grátis para compras acima de R$ 150,00.',
    'como falar com atendente humano':   'Digite 0 a qualquer momento para transferir ao atendente.',
}

def similaridade(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def buscar_resposta(pergunta: str, limiar: float = 0.80) -> str:
    melhor_score = 0.0
    melhor_resposta = None

    for faq_pergunta, faq_resposta in FAQ.items():
        score = similaridade(pergunta, faq_pergunta)
        if score > melhor_score:
            melhor_score = score
            melhor_resposta = faq_resposta

    print(f'  [DEBUG] Melhor score de similaridade: {melhor_score:.2f}')

    if melhor_score >= limiar:
        return melhor_resposta
    return 'Desculpe, não encontrei uma resposta para isso. Tente reformular.'

def chatbot_faq():
    print('=== Central de Atendimento — FAQ Inteligente ===')
    print('Faça sua pergunta ou digite "sair".')
    print()
    while True:
        pergunta = input('Você: ').strip()
        if pergunta.lower() in ('sair', 'tchau', 'bye'):
            print('Bot: Atendimento encerrado. Obrigado!')
            break
        resposta = buscar_resposta(pergunta)
        print(f'Bot: {resposta}')
        print()

chatbot_faq()

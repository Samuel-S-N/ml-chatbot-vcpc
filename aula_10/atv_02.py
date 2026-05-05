  # ============================================================
  # EXERCÍCIO 2 — Chatbot Pattern Matching com Regex
  # Modelo: Suporte técnico básico
  # ============================================================
import re

# Base de padrões: (padrão_regex, resposta)
PADROES = [
    (r'\b(oi|olá|ola|bom dia|boa tarde|boa noite|hey|hello)\b',
     'Olá! Bem-vindo ao suporte técnico. Como posso ajudar?'),

    (r'\b(senha|password|login|acesso)\b',
     'Para redefinir sua senha, acesse: configuracoes > seguranca > redefinir senha.'),

    (r'\b(lento|travando|devagar|lag|performance)\b',
     'Tente limpar o cache do navegador e reiniciar o sistema.'),

    (r'\b(erro|error|bug|falha|problema)\b',
     'Pode descrever o erro com mais detalhes? Qual mensagem aparece na tela?'),

    (r'\b(obrigad[ao]|valeu|thanks|thank you)\b',
     'Fico feliz em ajudar! Há mais alguma coisa?'),

    (r'\b(fatura|pagamento)\b',
    'As faturas são geradas no dia 15 de cada mês. O pagamento deve ser feito até o dia 20.'),

    (r'\b(tchau|sair|encerrar|bye|adeus)\b',
     'ENCERRAR'),
]

def responder(mensagem: str) -> str:
    mensagem_lower = mensagem.lower()
    for padrao, resposta in PADROES:
        if re.search(padrao, mensagem_lower):
            return resposta
    return 'Não encontrei informações sobre isso. Poderia reformular a pergunta?'

def chatbot_suporte():
    print('=== Chatbot de Suporte Técnico v2.0 ===')
    print('Digite sua dúvida ou "tchau" para sair.')
    print()
    while True:
        entrada = input('Você: ').strip()
        if not entrada:
            continue
        resposta = responder(entrada)
        if resposta == 'ENCERRAR':
            print('Bot: Até logo! Chamado encerrado. 🎫')
            break
        print(f'Bot: {resposta}')
        print()

chatbot_suporte()
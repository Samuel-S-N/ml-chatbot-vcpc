# ============================================================
# EXERCÍCIO 4 — Chatbot com Memória de Contexto
# Você deve completar as funções marcadas com TODO
# ============================================================

from datetime import datetime

# Estrutura do histórico de conversa
historico = []  # Lista de dicionários {'turno': N, 'usuario': '...', 'bot': '...'} 

RESPOSTAS = {
    'nome':     'Meu nome é ByteBot, seu assistente digital!',
    'ajuda':    'Posso responder sobre: nome, turno, historico, hora.',
    'hora':     lambda: f'Agora são {datetime.now().strftime("%H:%M:%S")}.',
}

def processar_mensagem(mensagem: str, turno: int) -> str:
    msg = mensagem.lower().strip()

    if 'nome' in msg:
        return RESPOSTAS['nome']
    elif 'hora' in msg:
        return RESPOSTAS['hora']()
    elif 'turno' in msg:
        return f'Já ocorreram {turno} turno(s) nesta sessão.'
    elif 'historico' in msg or 'histórico' in msg:
        if not historico:
            return 'Ainda não há mensagens anteriores no histórico.'
        ultimas = historico[-3:]
        linhas = [f"  Turno {h['turno']}: \"{h['usuario']}\"" for h in ultimas]
        return 'Últimas 3 mensagens suas:\n' + '\n'.join(linhas)
    elif 'ajuda' in msg:
        return RESPOSTAS['ajuda']
    else:
        return 'Não entendi. Digite "ajuda" para ver o que posso fazer.'

def registrar_turno(turno: int, usuario: str, bot: str):
    historico.append({'turno': turno, 'usuario': usuario, 'bot': bot})

def chatbot_contextual():
    print('=== ByteBot — Chatbot com Memória ===')
    turno = 0
    while True:
        entrada = input('Você: ').strip()
        if entrada.lower() in ('sair', 'tchau'):
            print('Bot: Até mais! Foram', turno, 'turnos de conversa.')
            break
        turno += 1
        resposta = processar_mensagem(entrada, turno)
        registrar_turno(turno, entrada, resposta)
        print(f'Bot: {resposta}')
        print()

chatbot_contextual()
#============================================================
# EXERCÍCIO 6 — Chatbot Multi-Personalidade
# Padrão avançado: troca de comportamento em runtime
# ============================================================
import re


class Personalidade:
    def __init__(self, nome: str, tom: str, prefixo_resposta: str, vocabulario_proibido: list):
        self.nome = nome
        self.tom = tom
        self.prefixo_resposta = prefixo_resposta
        self.vocabulario_proibido = vocabulario_proibido


# Defina pelo menos 3 personalidades:
PERSONALIDADES = {
    'formal': Personalidade(
        nome='Assistente Formal',
        tom='cordial e profissional',
        prefixo_resposta='Prezado usuário, ',
        vocabulario_proibido=['cara', 'mano', 'oi', 'beleza', 'vlw'],
    ),
    'casual': Personalidade(
        nome='ChatZinho',
        tom='descontraído e jovem',
        prefixo_resposta='Oi! ',
        vocabulario_proibido=['prezado', 'solicito', 'outrossim', 'conforme'],
    ),
    'tecnico': Personalidade(
        nome='TechBot',
        tom='técnico e direto',
        prefixo_resposta='RESPOSTA: ',
        vocabulario_proibido=['acho', 'talvez', 'quem sabe'],
    ),
}

RESPOSTAS_GENERICAS = {
    'python': 'Python é uma linguagem de programação interpretada e de alto nível.',
    'chatbot': 'Chatbot é um sistema que simula conversas humanas de forma automatizada.',
    'ia': 'Inteligência Artificial é a capacidade de máquinas realizarem tarefas cognitivas.',
    'ajuda': 'Posso falar sobre: python, chatbot, ia. Troque personalidade com /modo.',
}


def _substituir_vocabulario_proibido(texto: str, proibidos: list) -> str:
    resultado = texto
    for palavra in proibidos:
        padrao = re.compile(r'\b' + re.escape(palavra) + r'\b', re.IGNORECASE)
        resultado = padrao.sub('***', resultado)
    return resultado


class ChatbotMultiPersonalidade:
    def __init__(self):
        self.personalidade_ativa = PERSONALIDADES['formal']
        self.historico = []

    def trocar_personalidade(self, nome: str) -> str:
        chave = nome.strip().lower()
        if chave not in PERSONALIDADES:
            return (
                f'Modo "{nome}" não existe. Use: '
                + ', '.join(sorted(PERSONALIDADES.keys()))
                + '.'
            )
        self.personalidade_ativa = PERSONALIDADES[chave]
        return f'Personalidade alterada para: {self.personalidade_ativa.nome} ({self.personalidade_ativa.tom}).'

    def gerar_resposta(self, mensagem: str) -> str:
        msg = mensagem.lower()
        base = None
        for chave, texto in RESPOSTAS_GENERICAS.items():
            if chave in msg:
                base = texto
                break
        if base is None:
            base = (
                'Não reconheci o tema. Posso ajudar com: python, chatbot, ia. '
                'Digite ajuda para ver os comandos.'
            )

        # Tom: neste simulador, o prefixo e o vocabulário refletem o tom configurado.
        resposta = self.personalidade_ativa.prefixo_resposta + base
        resposta = _substituir_vocabulario_proibido(
            resposta, self.personalidade_ativa.vocabulario_proibido
        )
        self.historico.append((mensagem, resposta))
        return resposta

    def executar(self):
        print('=== Chatbot Multi-Personalidade ===')
        print('Comandos: /modo formal | /modo casual | /modo tecnico | sair')
        print(f'Personalidade ativa: {self.personalidade_ativa.nome}')
        print()
        while True:
            entrada = input('Você: ').strip()
            if entrada.lower() == 'sair':
                print('Encerrando. Até logo!')
                break
            elif entrada.lower().startswith('/modo '):
                nome = entrada.split(' ', 1)[1].lower()
                print(f'Bot: {self.trocar_personalidade(nome)}')
            else:
                resposta = self.gerar_resposta(entrada)
                print(f'Bot: {resposta}')
            print()


if __name__ == '__main__':
    ChatbotMultiPersonalidade().executar()

# ------------------------------------------------------------
# REFLEXÃO — Relação com "System Prompt" em LLMs (ex.: Claude)
# ------------------------------------------------------------
# Em um LLM, o system prompt é uma instrução persistente que define papel,
# tom, restrições e vocabulário antes da conversa do usuário. Aqui,
# Personalidade + prefixo_resposta + vocabulario_proibido funcionam como um
# "mini system prompt": ao chamar trocar_personalidade(), você troca esse
# bloco de instruções em runtime — o mesmo padrão de prompt-switching ou
# roteamento por persona em aplicações reais, só que explícito no código.

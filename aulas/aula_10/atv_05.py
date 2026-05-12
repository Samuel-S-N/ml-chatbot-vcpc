#============================================================
# EXERCÍCIO 5 — Intent + Entity Extraction
# Chatbot de Pedido de Pizza
# ============================================================
import re

SABORES   = ['calabresa', 'frango', 'queijo', 'portuguesa', 'vegetariana', 'pepperoni']
TAMANHOS  = ['pequena', 'media', 'média', 'grande', 'gigante', 'família', 'familia']

def detectar_intencao(mensagem: str) -> str:
    msg = mensagem.lower()
    if any(p in msg for p in ['quero', 'pedir', 'pedido', 'comprar', 'queria']):
        return 'FAZER_PEDIDO'
    # pedido direto: "pizza média de frango" (sem "quero")
    if 'pizza' in msg and any(t in msg for t in TAMANHOS):
        return 'FAZER_PEDIDO'
    elif any(p in msg for p in ['cancelar', 'desistir', 'não quero']):
        return 'CANCELAR'
    elif any(p in msg for p in ['cardápio', 'cardapio', 'opções', 'opcoes', 'tem']):
        return 'VER_CARDAPIO'
    return 'DESCONHECIDO'

def extrair_entidades(mensagem: str) -> dict:
    msg = mensagem.lower()
    sabor = next((s for s in SABORES if s in msg), None)
    tamanho = next((t for t in TAMANHOS if t in msg), None)
    if tamanho == 'media':
        tamanho = 'média'
    if tamanho == 'familia':
        tamanho = 'família'
    return {'sabor': sabor, 'tamanho': tamanho}

def confirmar_pedido(entidades: dict) -> str:
    sabor = entidades.get('sabor')
    tamanho = entidades.get('tamanho')
    if sabor and tamanho:
        return f'Pedido confirmado: pizza {tamanho} de {sabor}. Bom apetite!'
    if not sabor and not tamanho:
        return 'Não identifiquei sabor nem tamanho. Diga, por exemplo: pizza grande de calabresa.'
    if not sabor:
        return f'Qual sabor você quer na pizza {tamanho}? Sabores: {", ".join(SABORES)}.'
    return 'Qual tamanho? Opções: pequena, média, grande, gigante ou família.'

def chatbot_pizza():
    print('=== PizzaBot — Faça seu pedido! ===')
    while True:
        entrada = input('Você: ').strip()
        if entrada.lower() in ('sair', 'tchau'):
            print('Bot: Pedido cancelado. Até logo!')
            break

        intencao = detectar_intencao(entrada)
        print(f'  [DEBUG] Intenção: {intencao}')

        if intencao == 'VER_CARDAPIO':
            print(f'Bot: Sabores disponíveis: {", ".join(SABORES)}')
            print(f'Bot: Tamanhos: {", ".join(["Pequena", "Média", "Grande"])}')
        elif intencao == 'FAZER_PEDIDO':
            entidades = extrair_entidades(entrada)
            print(f'  [DEBUG] Entidades: {entidades}')
            resposta = confirmar_pedido(entidades)
            print(f'Bot: {resposta}')
        elif intencao == 'CANCELAR':
            print('Bot: Seu pedido foi cancelado.')
        else:
            print('Bot: Não entendi. Você pode pedir uma pizza ou ver o cardápio.')
        print()

chatbot_pizza()

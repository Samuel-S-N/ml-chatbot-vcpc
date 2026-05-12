# ============================================================
  # EXERCÍCIO 1 — Chatbot Rule-Based (IF-ELSE)
  # Modelo: Atendente de Lanchonete
  # ============================================================
  
def chatbot_lanchonete():
  print('Como posso ajudar você hoje?')
  print('=== Bem-vindo à Lanchonete Digital! ===')
  print('Opções: [cardapio] [pedido] [preco] [sair]')
  print()

  cardapio = {
      'hamburguer': 25.90,
      'pizza':      18.50,
      'suco':        8.00,
      'sorvete':    12.00
  }

  while True:
      entrada = input('Você: ').strip().lower()

      if entrada == 'cardapio':
          print('Bot: Nosso cardápio:')
          for item, preco in cardapio.items():
              print(f'       {item.capitalize()} — R$ {preco:.2f}')

      elif entrada == 'pedido':
          item = input('Bot: Qual item você deseja? ').strip().lower()
          if item in cardapio:
              print(f'Bot: Perfeito! {item.capitalize()} adicionado. Total: R$ {cardapio[item]:.2f}')
          else:
              print('Bot: Desculpe, esse item não está no cardápio.')

      elif entrada == 'preco':
          item = input('Bot: Preço de qual item? ').strip().lower()
          if item in cardapio:
              print(f'Bot: {item.capitalize()} custa R$ {cardapio[item]:.2f}.')
          else:
              print('Bot: Item não encontrado no cardápio.')

      elif entrada == 'sair':
          print('Bot: Obrigado pela visita! Até logo! 👋')
          break

      else:
          print('Bot: Não entendi. Tente: [cardapio] [pedido] [preco] [sair]')
  
  # Executar o chatbot
chatbot_lanchonete()
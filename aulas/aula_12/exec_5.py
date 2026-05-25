# Exercício 5: Máquina de Estados Finita (FSM) Guiada por Menu Interativo
# ORIENTAÇÃO: Projete um bot estruturado em árvore de decisões onde o input do usuário
#   dita as transições de estados válidos do sistema (Menu -> Suporte -> Pagamento).
# OBJETIVO: Dominar o conceito de FSM (Finite State Machine) para garantir que o usuário
#   navegue em fluxos lógicos protegidos contra inputs inválidos.
# TESTE: Digite 3 mensagens e colete as respostas

SINAL_ENCERRAR = "ENCERRAR"

MENSAGEM_MENU = (
    "Bem-vindo ao [MENU_PRINCIPAL].\n"
    "  1 — Suporte (pedidos, entregas, reclamações)\n"
    "  2 — Financeiro / Pagamento (boletos, estornos, faturas)\n"
    "  3 — Encerrar atendimento\n"
    "Digite o número da opção desejada."
)

MENSAGEM_SUPORTE = (
    "Modo [SUPORTE] ativo.\n"
    "  1 — Rastrear pedido\n"
    "  2 — Abrir reclamação\n"
    "  0 — Voltar ao menu principal\n"
    "  9 — Encerrar atendimento"
)

MENSAGEM_FINANCEIRO = (
    "Modo [FINANCEIRO / PAGAMENTO] ativo.\n"
    "  1 — Consultar boleto ou fatura\n"
    "  2 — Solicitar estorno\n"
    "  0 — Voltar ao menu principal\n"
    "  9 — Encerrar atendimento"
)


class BotFSM:
    """Máquina de estados: MENU_PRINCIPAL → SUPORTE | FINANCEIRO → sub-ações."""

    def __init__(self):
        self.estado_atual = "MENU_PRINCIPAL"

    def _voltar_menu(self) -> str:
        self.estado_atual = "MENU_PRINCIPAL"
        return "Retornando ao [MENU_PRINCIPAL].\n" + MENSAGEM_MENU.split("\n", 1)[1]

    def _encerrar(self) -> str:
        self.estado_atual = "ENCERRADO"
        return SINAL_ENCERRAR

    def transicionar(self, opcao: str) -> str:
        opcao = opcao.strip().lower()

        if opcao in ("sair", "encerrar", "tchau"):
            return self._encerrar()

        if self.estado_atual == "MENU_PRINCIPAL":
            if opcao == "1":
                self.estado_atual = "SUPORTE"
                return MENSAGEM_SUPORTE
            if opcao == "2":
                self.estado_atual = "FINANCEIRO"
                return MENSAGEM_FINANCEIRO
            if opcao == "3":
                return self._encerrar()
            return (
                "Opção inválida. No menu principal, escolha "
                "1 (Suporte), 2 (Financeiro) ou 3 (Encerrar)."
            )

        if opcao == "9":
            return self._encerrar()

        if opcao == "0":
            return self._voltar_menu()

        if self.estado_atual == "SUPORTE":
            if opcao == "1":
                return (
                    "Informe o número do pedido (formato #1234) na próxima mensagem "
                    "ou digite 0 para voltar."
                )
            if opcao == "2":
                return (
                    "Reclamação registrada em fila de atendimento. "
                    "Descreva o problema em detalhes ou digite 0 para voltar."
                )
            return (
                "Opção inválida em [SUPORTE]. "
                "Use 1 (rastreio), 2 (reclamação), 0 (menu) ou 9 (encerrar)."
            )

        if self.estado_atual == "FINANCEIRO":
            if opcao == "1":
                return (
                    "Consulta de boleto/fatura: envie o CPF ou número da fatura. "
                    "Digite 0 para voltar ao menu."
                )
            if opcao == "2":
                return (
                    "Estorno: informe o número do pedido (#NNNN). "
                    "Prazo de análise: até 5 dias úteis. Digite 0 para voltar."
                )
            return (
                "Opção inválida em [FINANCEIRO]. "
                "Use 1 (boleto), 2 (estorno), 0 (menu) ou 9 (encerrar)."
            )

        return f"Estado desconhecido: {self.estado_atual}."


def iniciar_chatbot():
    print("=" * 55)
    print("  ASSISTENTE E-COMMERCE — Exercício 05")
    print("  Máquina de Estados Finita (FSM)")
    print("=" * 55)
    print("  Fluxo: Menu → Suporte ou Financeiro (Pagamento).")
    print("  Encerre com a opção 3 (menu), 9 (submenus) ou digitando 'sair'.\n")

    bot = BotFSM()
    print(f"Bot: {MENSAGEM_MENU}")
    turno = 0

    while True:
        try:
            entrada = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Sistema encerrado]")
            break

        if not entrada:
            print("Bot: Por favor, digite uma opção válida.\n")
            continue

        turno += 1
        resposta = bot.transicionar(entrada)

        if resposta == SINAL_ENCERRAR:
            print("Bot: Obrigado pelo contato! Atendimento encerrado. Até logo!")
            print(f"     [estado final: {bot.estado_atual} | interações: {turno}]\n")
            break

        print(f"Bot: {resposta}")
        print(f"     [estado atual: {bot.estado_atual}]\n")


if __name__ == "__main__":
    iniciar_chatbot()

# Exercício 9: Integração com a API do Google Gemini (System Instructions via Prompt)
# ORIENTAÇÃO: Crie um loop de conversação real integrado à API do Gemini. Configure o modelo
#   para agir sob uma persona de especialista e use input() para enviar os prompts.
# OBJETIVO: Aprender a instanciar o cliente oficial do Google GenAI, tratar variáveis de
#   ambiente para a API Key e customizar comportamentos via system_instruction.
# TESTE: coloque o resultado de saída

import os

from google import genai
from google.genai import types

# Chave configurada para o exercício (prioridade: variável já definida no ambiente)
if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = "AIzaSyBqrwljdxe1EGW2RkBHsINVqoVrsgJoSK4"

MODELO = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = (
    "Você é o mestre dos magos de um RPG de TI. "
    "Fale de forma mística e curta. Responda dúvidas técnicas sobre "
    "programação, dados e sistemas como se fossem feitiços e artefatos."
)


class ChatbotGemini:
    """Chat com persona fixa via system_instruction na API Google GenAI."""

    def __init__(
        self,
        api_key: str | None = None,
        modelo: str = MODELO,
        system_instruction: str = SYSTEM_INSTRUCTION,
        temperature: float = 0.7,
    ):
        chave = api_key or os.environ.get("GEMINI_API_KEY")
        if not chave:
            raise ValueError(
                "API Key não encontrada. Defina GEMINI_API_KEY no ambiente."
            )

        self.client = genai.Client(api_key=chave)
        self.modelo = modelo
        self.config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

    def perguntar(self, mensagem: str) -> str:
        resposta = self.client.models.generate_content(
            model=self.modelo,
            contents=mensagem,
            config=self.config,
        )
        return resposta.text or "(Resposta vazia do modelo.)"


def iniciar_chatbot():
    print("=" * 55)
    print("  ASSISTENTE E-COMMERCE — Exercício 09")
    print("  Google Gemini + System Instruction")
    print("=" * 55)
    print("  Persona: Mestre dos Magos do RPG de TI.")
    print("  Digite 'sair' para encerrar.\n")

    try:
        bot = ChatbotGemini()
    except ValueError as e:
        print(f"Bot: {e}")
        return

    print(
        "Bot: Iniciando conexão com o ecossistema Google AI. "
        "Faça sua pergunta técnica:"
    )
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
            print(f"Bot: Até logo, viajante! Foram {turno} perguntas ao oráculo.")
            break

        turno += 1
        try:
            texto = bot.perguntar(entrada)
            print(f"Bot: {texto}\n")
        except Exception as e:
            print(
                "Bot: Erro de conexão ou API Key inválida. "
                f"Verifique as configurações. Descrição: {e}\n"
            )


if __name__ == "__main__":
    iniciar_chatbot()

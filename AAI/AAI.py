"""
Pergunta 2 - Filtro de NLP com Ponderação de Stopwords Customizadas
Chatbot de suporte técnico: preserva "não", "erro", "falha" e o símbolo %
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)


def heavy_clean_nlp(text: str) -> list[str]:
    
    text = text.lower()
    
    text = text.replace('_', ' ')
    text = re.sub(r'[^\w%]+', ' ', text)
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('portuguese'))
    custom_stopwords = stop_words - {'não', 'erro', 'falha'}
    
    tokens_limpos = [t for t in tokens if t not in custom_stopwords and t.strip()]
    
    return sorted(list(set(tokens_limpos)))


if __name__ == "__main__":
    texto_teste = "O sinal NÃO está em 100%!!! Erro de conexão ou falha_técnica?"
    
    resultado = heavy_clean_nlp(texto_teste)
    
    print("=" * 60)
    print("TESTE: heavy_clean_nlp()")
    print("=" * 60)
    print(f"Entrada:  {texto_teste}")
    print(f"Output:   {resultado}")

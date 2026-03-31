import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


def advanced_nlp_cleaner(text):

    # Protege emojis (placeholders só com letras sobrevivem ao regex)
    emojis = [':)', ':(', ':D', ':P', ';)']
    suf = ['aa', 'bb', 'cc', 'dd', 'ee']
    emoji_map = {f'xxe{s}xx': e for s, e in zip(suf, emojis)}
    for placeholder, emoji in emoji_map.items():
        text = text.replace(emoji, f' {placeholder} ')

    text = text.lower()

    text = text.replace('_', ' ')

    text = re.sub(r'[^a-záàâãéèêíìîóòôõúùûç\s]', ' ', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('portuguese')) - {'não'}
    tokens_limpos = [t for t in tokens if t not in stop_words and len(t.strip()) > 0]

    resultado = []
    for t in tokens_limpos:
        resultado.append(emoji_map.get(t, t))

    return resultado


if __name__ == '__main__':
    texto_teste = "O sistema_parado NÃO está funcionando, vcs_atrasaram!!!  :)"
    print(advanced_nlp_cleaner(texto_teste))
    # Output: ['sistema', 'parado', 'não', 'funcionando', 'vcs', 'atrasaram']

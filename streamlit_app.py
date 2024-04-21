import streamlit as st
from transformers import MarianTokenizer, MarianMTModel

@st.cache
def load_model_and_tokenizer():
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    return tokenizer, model

def translate_to_spanish(input_text, max_length=None):
    tokenizer, model = load_model_and_tokenizer()

    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(input_text)))
    num_tokens = len(tokens)

    try:
        if not max_length:
            max_length = num_tokens + 50

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        translated_ids = model.generate(input_ids, max_length=max_length, num_beams=4, length_penalty=2.0)

        translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

        return translated_text
    except IndexError:
        raise ValueError(f"El texto de entrada debe tener un mínimo de 3 palabras y un máximo aproximado de {max_length} palabras.")

def main():
    st.title("Traductor de Español a Inglés")

    text = st.text_area("Introduce el texto en español que quieres traducir:")
    if st.button("Traducir"):
        input_text = text.lstrip("¿¡")
        try:
            translation = translate_to_spanish(input_text, max_length=5000)
            st.success("Traducción:")
            st.write(translation)
        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

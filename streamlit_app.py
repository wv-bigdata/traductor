import streamlit as st
from transformers import MarianTokenizer, MarianMTModel

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model_and_tokenizer_es_en():
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    return tokenizer, model

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model_and_tokenizer_en_es():
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    return tokenizer, model

def translate_text(input_text, model_name, max_length=None):
    tokenizer, model = model_name()

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
    st.title("Traductor Español - Inglés / Inglés - Español")

    source_lang = st.selectbox("Selecciona el idioma de origen:", ["Español", "Inglés"])
    target_lang = "Inglés" if source_lang == "Español" else "Español"
    model_name = load_model_and_tokenizer_es_en if source_lang == "Español" else load_model_and_tokenizer_en_es
    
    text = st.text_area(f"Introduce el texto en {source_lang} que quieres traducir:")
    
    if st.button("Traducir"):
        input_text = text.strip("¿¡")  # Eliminar caracteres especiales del principio
        try:
            translation = translate_text(input_text, model_name, max_length=5000)
            st.success(f"Traducción al {target_lang}:")
            max_chars_per_line = 80  # Límite de caracteres por línea
            start = 0
            while start < len(translation):
                end = start + max_chars_per_line
                st.markdown(f"```\n{translation[start:end]}\n```")
                start = end
        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

# Pie de página
st.caption("Traductor de idiomas desarrollado por Wilbert Vong - Big Data Architect.")

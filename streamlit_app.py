import streamlit as st
from transformers import MarianTokenizer, MarianMTModel

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model_and_tokenizer(source_lang, target_lang):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(input_text, source_lang, target_lang, max_length=None):
    tokenizer, model = load_model_and_tokenizer(source_lang, target_lang)

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
    st.title("Traductor")

    source_lang = st.selectbox("Selecciona el idioma de origen:", ["español", "inglés"])
    target_lang = "inglés" if source_lang == "español" else "español"
    
    text = st.text_area(f"Introduce el texto en {source_lang} que quieres traducir:")
    
    if st.button("Traducir"):
        input_text = text.strip("¿¡")  # Eliminar caracteres especiales del principio
        try:
            translation = translate_text(input_text, source_lang[:2], target_lang[:2], max_length=5000)
            st.success(f"Traducción a {target_lang}:")
            st.write(translation)
        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

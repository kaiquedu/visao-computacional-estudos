import streamlit as st

from classifier import load_model, predict

EMOJIS = {"Gato": "😺", "Cachorro": "🐶"}

st.set_page_config(page_title="Classificação de Cachorros e Gatos 🐶🐱")
st.title("Classificação de Cachorros e Gatos 🐶🐱")
st.write("Envie uma imagem para que o modelo classifique se é um **cachorro** ou um **gato**.")


@st.cache_resource(show_spinner="Carregando modelo...")
def get_model():
    return load_model()


uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagem carregada", use_container_width=True)
    with st.spinner("Classificando..."):
        result = predict(get_model(), uploaded_file)
    if result.is_confident:
        st.success(f"Resultado: **{result.label} {EMOJIS[result.label]}** ({result.confidence:.0%} de confiança)")
    else:
        st.warning("Não tenho certeza... pode ser outro animal! 🤔")

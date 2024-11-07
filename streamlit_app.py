import streamlit as st
import transformers
# st.write('Wykorzystaj Huggin Face do stworzenia swojej własnej aplikacji tłumaczącej tekst z języka angielskiego na język niemiecki. Zmodyfikuj powyższy kod dodając do niego kolejną opcję, tj. tłumaczenie tekstu. Informacje potrzebne do zmodyfikowania kodu znajdziesz na stronie Huggin Face - https://huggingface.co/docs/transformers/index')
# st.write('🐞 Dodaj właściwy tytuł do swojej aplikacji, może jakieś grafiki?')
# st.write('🐞 Dodaj krótką instrukcję i napisz do czego służy aplikacja')
# st.write('🐞 Wpłyń na user experience, dodaj informacje o ładowaniu, sukcesie, błędzie, itd.')
# st.write('🐞 Na końcu umieść swój numer indeksu')
# st.write('🐞 Stwórz nowe repozytorium na GitHub, dodaj do niego swoją aplikację, plik z wymaganiami (requirements.txt)')
# st.write('🐞 Udostępnij stworzoną przez siebie aplikację (https://share.streamlit.io) a link prześlij do prowadzącego')

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
st.header('Tłumacz')

st.subheader('Tłumaczenie języka z angielskiego na niemiecki')
transformers.T5Config(use_cache=False)

success = [None]
def AcceptTextGer(text):
    input_ids = tokenizer("translate English to German: " + text, return_tensors="pt").input_ids
    if text:
        if success[0] is not None:
            success[0].empty()
        info = st.info('W trakcie')
        outputs = model.generate(input_ids)
        st.write(tokenizer.decode(outputs[0], skip_special_tokens=True))
        info.empty()
        success[0] = st.success('Udało się!')


textG = st.text_area(label="Wpisz tekst w pole by przetłumaczyć na niemiecki")
st.text('Wynik tłumaczenia:')
st.button("Translate to german", on_click=AcceptTextGer(textG))
st.text('Projekt stworzony przez s24600:')

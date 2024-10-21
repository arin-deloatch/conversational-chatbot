import streamlit as st
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv() 


class QACompanion:
    def __init__(self, MODEL_PATH: str, TOKENIZER_PATH: str) -> None:
        self.title = 'QA Companion 🤖'
        self.model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)
        self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

        HF_ACCESS_TOKEN = os.environ.get('HF_ACCESS_TOKEN')

        self.qa = pipeline("question-answering", model=self.model,
                                         tokenizer=self.tokenizer,
                                         token=os.environ.get('HF_ACCESS_TOKEN'))
        
    def app(self) -> None:
        st.title(self.title)

        question = st.text_input("Ask QA Companion a question:")
        context = st.text_input("Enter the context of your question:")

        if st.button("Generate Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    answer = self.qa(question,context)
                    st.write("### Answer:")
                    st.write(answer)
            else:
                st.warning("Please enter a question.")


if __name__ == '__main__':
    MODEL_PATH = 'pseoul/bert-cased-qa-companion'
    TOKENIZER_PATH = 'pseoul/bert-cased-qa-companion'
    bot = QACompanion(MODEL_PATH=MODEL_PATH, TOKENIZER_PATH=TOKENIZER_PATH)
    bot.app()
    
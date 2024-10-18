import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import pipeline


class QACompanion:
    def __init__(self, MODEL_PATH: str, TOKENIZER_PATH: str) -> None:
        self.title = 'QA Companion 🤖'
        self.model = DistilBertForQuestionAnswering.from_pretrained(MODEL_PATH)
        self.tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)
      
        
    def generate_answer(self, question: str, context:str):
        try:
            question_answerer = pipeline("question-answering", model=self.model,
                                         tokenizer=self.tokenizer)
            question_answerer(question=question, context=context)
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def app(self) -> None:
        st.title(self.title)

        question = st.text_input("Ask QA Companion a question:")
        context = st.text_input("Enter the context of your question:")

        if st.button("Generate Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    answer = self.generate_answer(question,context)
                    st.write("### Answer:")
                    st.write(answer)
            else:
                st.warning("Please enter a question.")


if __name__ == '__main__':
    '''
    TODO: Replace the MODEL_PATH AND TOKENIZER_PATH with our fine-tuned model
    '''
    MODEL_PATH = 'distilbert-base-uncased'
    TOKENIZER_PATH = 'distilbert-base-uncased'
    bot = QACompanion(MODEL_PATH=MODEL_PATH, TOKENIZER_PATH=TOKENIZER_PATH)
    bot.app()
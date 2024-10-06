import streamlit as st
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

'''
TODO: 
1. Add model and tokenizer caching mechanisms to prevent OOM kills
2. Update interface to resemble that of a constant dialogue between user and system
3. Replace BART with fine-tuned BART

'''
class ConversationalCompanion:
    def __init__(self,MODEL_PATH:str, TOKENIZER_PATH:str) -> None:
        self.title = 'Conversational Companion :robot_face:'
        self.model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
        self.tokenizer = BartTokenizer.from_pretrained(TOKENIZER_PATH)

    def generate_answer(self, question: str):
        input_text = f"question: {question}"
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)

        outputs = self.model.generate(inputs['input_ids'])
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def app(self) -> None:
        st.title(self.title)

        question = st.text_input("Ask Conversational Companion a question:")

        if st.button("Generate Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    answer = self.generate_answer(question)
                    st.write("### Answer:")
                    st.write(answer)
            else:
                st.warning("Please enter a question.")



if __name__ == '__main__':
    MODEL_PATH = 'facebook/bart-large' # Replace when fine-tuning is complete
    TOKENIZER_PATH = 'facebook/bart-large' # Replace when fine-tuning is complete
    bot = ConversationalCompanion(MODEL_PATH=MODEL_PATH,TOKENIZER_PATH=TOKENIZER_PATH)
    bot.app()
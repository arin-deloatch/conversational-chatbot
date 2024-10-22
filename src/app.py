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
    def predictTextAnswers(self, prompts, model, tokenizer):
    # returns the predicted answer (text) given the context and question provided in the prompt.
    # prompt is a sequence of dicts (can be of length 1).
    # Each dict must contain the keys 'question' and 'answer', and may contain others
    # If the key 'id' is provided, it will be returned (needed for evaluation function)

        question = prompts['question']
        context = prompts['context']

        # use GPU if available
        if torch.backends.mps.is_available():
            print("MPS device found.")
            torch.mps.empty_cache()
            device = torch.device("mps")
        else:
            print("MPS device not found.")
            device = torch.device("cpu")
        model.to(device)

        question = prompts['question']
        context = prompts['context']

        # tokenize prompt
        inputs = tokenizer(
            question,
            context,
            max_length=384,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt"
        ).to(device)


        model.eval()

        with torch.no_grad():
            outputs = model(**inputs)

        # Use positions with max probabilities
        answer_start = torch.argmax(outputs.start_logits, dim=-1)
        answer_end = torch.argmax(outputs.end_logits, dim=-1)

        # Convert token positions into text spans for each sample in the batch
        answer_texts = []
        for i, input_ids in enumerate(inputs['input_ids']):
            answer_text = tokenizer.convert_tokens_to_string(
                            tokenizer.convert_ids_to_tokens(input_ids[answer_start[i]:answer_end[i]+1]))
            answer_texts.append(answer_text)

        # include 'id' key in the return value if it exists in the input prompt
        predicted_answer = {}
        if 'id' in prompts:
            predicted_answer['id'] = prompts['id']
        predicted_answer['prediction_text'] = answer_texts

        return predicted_answer  
        
    def prettyPrintSingleShotQA(self,prompt, model, tokenizer):
        pred_answer = self.predictTextAnswers(prompt, model, tokenizer)['prediction_text'][0]
        # print Context, Question, Answer
        print(f"CONTEXT:\n{prompt['context']}\n")
        print(f"QUESTION:\n{prompt['question']}\n")
        print(f"MODEL ANSWER:\n{pred_answer}\n")

        return pred_answer
        
    def app(self) -> None:
        st.title(self.title)

        question = st.text_area("Ask QA Companion a question:")
        context = st.text_area("Enter the context of your question:")

        prompt = {
            'question': question,
            'context' : context
        }

        if st.button("Generate Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    answer = self.prettyPrintSingleShotQA(prompt,self.model,self.tokenizer)
                    st.write("### Answer:")
                    st.write(answer)
            else:
                st.warning("Please enter a question.")



if __name__ == '__main__':
    MODEL_PATH = 'pseoul/bert-cased-qa-companion'
    TOKENIZER_PATH = 'pseoul/bert-cased-qa-companion'
    bot = QACompanion(MODEL_PATH=MODEL_PATH, TOKENIZER_PATH=TOKENIZER_PATH)
    bot.app()
    

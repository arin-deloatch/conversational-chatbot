import streamlit as st

class ConversationalCompanion:
    def __init__(self) -> None:
        self.title = 'Conversational Companion :robot_face:'

    def app(self) -> None:
        st.title(self.title)
        prompt = st.chat_input("Ask Conversational Companion a question!")

        if prompt:
            st.write(f"User has sent the following prompt: {prompt}")


if __name__ == '__main__':
    bot = ConversationalCompanion()
    bot.app()
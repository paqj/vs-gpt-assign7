import streamlit as st

st.title("Input Your OpenAI API Key on Side Bar")

with st.sidebar:
    # st.title("Input Your OpenAI API Key on Side Bar")
    # 사용자로부터 OpenAI API 키 입력 받기
    user_api_key = st.text_input("Enter your OpenAI API key", "")

    st.markdown(
        """
        [GitHub Repository](https://github.com/paqj/vs-gpt-assign5)
        """,
    )
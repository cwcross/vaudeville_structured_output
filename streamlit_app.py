import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("Vaudeville play: structured output of musical moments")
st.write(
    "Upload a document below and an OpenAI model will return a csv file that you can import to excel or google sheets. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). Or, you can enter the passcode we provided to use ours."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
password = st.text_input("Password", type="password")
if not openai_api_key and not password:
    st.info("Please add your OpenAI API key or the password we sent to continue.", icon="🗝️")
else:

    entry_granted: bool = False

    if openai_api_key:
    # Create an OpenAI client.
        client = OpenAI(api_key=openai_api_key)
        entry_granted = True

    if password


if not entry_granted:
    

else:

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)

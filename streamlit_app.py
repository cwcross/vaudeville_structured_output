import streamlit as st
from openai import OpenAI
import os
import io
import tempfile
import pandas as pd
import csv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Optional
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

# Styling
st.set_page_config(page_title="Vaudeville Analyzer", layout="wide")
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.2em;
        color: #f3f3f3;
    }
    .description {
        font-size: 1.1rem;
        color: #bbbbbb;
        margin-bottom: 2rem;
    }
    .section {
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 600;
    }
    .stButton>button, .stDownloadButton>button {
        height: 3em;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="main-title">Vaudeville play: structured output of musical moments</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="description">Upload a document below and an OpenAI model will return a CSV file that you can import to Excel or Google Sheets. '
    'To use this app, you need to provide an OpenAI API key, which you can get <a href="https://platform.openai.com/account/api-keys" target="_blank">here</a>. '
    'Or, you can enter the passcode we provided to use ours.</div>',
    unsafe_allow_html=True
)

# Auth
if "entry_granted" not in st.session_state:
    st.session_state.entry_granted = False

if not st.session_state.entry_granted:
    with st.form("auth_form"):
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Submit")

    if submitted:
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.session_state.entry_granted = True
            st.rerun()
        elif password and password == st.secrets["entry_password"]:
            os.environ["OPENAI_API_KEY"] = st.secrets["RF_API_KEY"]
            st.session_state.entry_granted = True
            st.rerun()
        else:
            st.error("Invalid API key or password.")

# Main app
else:
    st.markdown('<div class="section">Upload a document (.pdf)</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=("pdf"))

    if uploaded_file:
        def loadPDF() -> list:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            loader = PyPDFLoader(tmp_path)
            file = loader.load()
            os.remove(tmp_path)
            return file

        source = loadPDF()
        source_content = "".join([page.page_content for page in source])
        source_full = Document(page_content=source_content, metadata=source[0].metadata)

        # Models
        llm = init_chat_model("gpt-4o", model_provider="openai")
        processing_llm = init_chat_model("gpt-4o-mini", model_provider="openai")

        class Scene(BaseModel):
            act: int
            scene: int
            header: str

        class FullPlay(BaseModel):
            all_scenes: List[Scene]

        formatted_splitter_llm = processing_llm.with_structured_output(FullPlay)
        prompt = f"""
        The following is the full text of a French Vaudeville play. Your job is to identify every scene boundary.
        For each scene, return:
        - The act number (as it appears in the text)
        - The scene number (as it appears in the text)
        - The exact scene header line (copy it verbatim from the text)

        Return a list of objects like:
        {{"act": "...", "scene": "...", "header": "..."}}

        Do not attempt to count character indexes. Only return the scene headers as they appear in the text.

        Play Content: \n
        {source_full.page_content}
        """

        def split_up_play(doc):
            response = formatted_splitter_llm.invoke(prompt)
            return response

        all_indexes = split_up_play(source_full)

        all_splits = []
        scene_headers = all_indexes.all_scenes
        full_text = source_full.page_content
        prev_end_idx = 0
        for i, scene in enumerate(scene_headers):
            start_idx = full_text.find(scene.header, prev_end_idx)
            if start_idx == -1:
                raise ValueError(f"Scene header not found: {scene.header}")
            end_idx = full_text.find(scene_headers[i + 1].header, start_idx + len(scene.header)) if i + 1 < len(scene_headers) else len(full_text)
            scene_text = full_text[start_idx:end_idx]
            doc = Document(page_content=scene_text, metadata={"act": scene.act, "scene": scene.scene, "header": scene.header})
            all_splits.append(doc)
            prev_end_idx = end_idx

        class MusicalMoment(BaseModel):
            act: int
            scene: int
            number: int
            characters: list[str]
            dramatic_situation: str
            air_or_melodie: str
            poetic_text: str
            rhyme_scheme: str
            poetic_form: str
            end_of_line_accents: list[str]
            syllable_count_per_line: list[int]
            irregularities: Optional[str]
            stage_direction_or_cues: Optional[str]
            reprise: Optional[str]

        class VaudevillePlay(BaseModel):
            musicalMoments: list[MusicalMoment]

        structured_llm = llm.with_structured_output(VaudevillePlay)
        system_prompt = """You are a literary analyst..."""
        human_prompt = """Given the following chunk..."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ])

        class State(TypedDict):
            index: int
            context: Document
            answer: str

        def check_index(state: State): return state
        def retrieve_doc(state: State): return {"context": all_splits[state["index"]]}
        def generate(state: State):
            i = state["index"]
            message = prompt.invoke({"question": human_prompt, "context": f'Act {all_indexes.all_scenes[i].act}, Scene {all_indexes.all_scenes[i].scene}:\n\n {state["context"].page_content}'})
            response = structured_llm.invoke(message)
            return {"answer": response}

        from langgraph.graph import START, StateGraph
        graph_builder = StateGraph(State).add_sequence([check_index, retrieve_doc, generate])
        graph_builder.add_edge(START, "check_index")
        graph = graph_builder.compile()

        def analyze_scenes(docs: List[Document]) -> List[MusicalMoment]:
            all_moments = []
            for i, doc in enumerate(docs):
                response = graph.invoke({"index": i})
                all_moments.extend(response["answer"].musicalMoments)
            return all_moments

        all_moments = analyze_scenes(all_splits)
        moments_dicts = [m.model_dump() for m in all_moments]

        # CSV
        output = io.StringIO()
        fieldnames = moments_dicts[0].keys() if moments_dicts else []
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in moments_dicts:
            for key, value in row.items():
                if isinstance(value, list):
                    row[key] = "; ".join(str(v) for v in value)
            writer.writerow(row)
        csv_bytes = output.getvalue().encode("utf-8")

        st.markdown('<div class="section">Download your CSV</div>', unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name="musical_moments.csv",
            mime="text/csv",
            use_container_width=True
        )

        df = pd.DataFrame(moments_dicts)
        st.markdown('<div class="section">🎼 Musical Moments Preview</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

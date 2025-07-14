import streamlit as st
from openai import OpenAI
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Optional
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
import io
import pandas as pd

# Show title and description.
st.set_page_config("centered")

st.title("Vaudeville play: structured output of musical moments")


# Initialize session state if it doesn't exist
if "entry_granted" not in st.session_state:
    st.session_state.entry_granted = False

# Show login form only if access hasn't been granted yet
if not st.session_state.entry_granted:
    with st.form("auth_form"):
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Submit")

    if submitted:
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.session_state.entry_granted = True
            st.success("Access granted using your API key.")
            st.rerun()
        elif password and password == st.secrets["entry_password"]:
            os.environ["OPENAI_API_KEY"] = st.secrets["RF_API_KEY"]
            st.session_state.entry_granted = True
            st.success("Access granted using the provided password.")
            st.rerun()
        else:
            st.error("Invalid API key or password.")
else:
    st.write(
    "Upload a document below and an OpenAI model will return a csv file that you can import to excel or google sheets. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "Or, you can enter the passcode we provided to use ours."
)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.pdf)", type=("pdf")
    )

    if uploaded_file:

        ##############
        #  PDF Loading
        ##############

        import tempfile

        def loadPDF() -> list:
            # Write uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            file = loader.load()
            os.remove(tmp_path)
            return file

        
        source = loadPDF()

        source_content = ""
        for page in source:
            source_content += page.page_content
        source_full = Document(page_content = source_content, metadata = source[0].metadata)

        # Security: Prevents large documents with our API Key.
        if len(source_full) > 100_000 and os.environ["OPENAI_API_KEY"] == st.secrets["RF_API_KEY"]:
            st.warning("Your file is over 100,000 characters. Please use your own API key.")
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
            

        #################################
        # Breaking up the PDF into scenes
        #################################


        llm = init_chat_model("gpt-4o", model_provider="openai")
        processing_llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        

        class Scene(BaseModel):
            """A single scene from a Vaudeville play"""

            act: int = Field(description="The act number or label as it appears in the text.")
            scene: int = Field(description="The scene number or label as it appears in the text.")
            header: str = Field(description="The exact scene header line, copied verbatim from the text.")

        class FullPlay(BaseModel):
            """A full play, that has yet to be broken into individual scenes."""

            all_scenes: List[Scene] = Field(description="A list of every single scene's header and label - each as a Scene object.")

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
        from langchain_core.documents import Document

        all_splits = []
        scene_headers = all_indexes.all_scenes
        full_text = source_full.page_content

        prev_end_idx = 0
        for i, scene in enumerate(scene_headers):
            # Find the start index of this scene's header after the previous end index
            start_idx = full_text.find(scene.header, prev_end_idx)
            if start_idx == -1:
                raise ValueError(f"Scene header not found: {scene.header}")

            # Determine the end index: start of next scene header, or end of document
            if i + 1 < len(scene_headers):
                next_start_idx = full_text.find(scene_headers[i + 1].header, start_idx + len(scene.header))
                if next_start_idx == -1:
                    end_idx = len(full_text)
                else:
                    end_idx = next_start_idx
            else:
                end_idx = len(full_text)

            scene_text = full_text[start_idx:end_idx]
            doc = Document(page_content=scene_text, metadata={"act": scene.act, "scene": scene.scene, "header": scene.header})
            all_splits.append(doc)
            prev_end_idx = end_idx

        ############################
        # Pydantic and LangGraph
        ############################


        # Pydantic
        class MusicalMoment(BaseModel):
            """Many of these musical moments reuse some preexisting (and often well-known)  melody or tune.  These are variously called "melodie”, or “air”, and identified with a short title that refers in some way to an opera or collection of melodies from which it was drawn.  The titles might include the names of works, or other characters in those original works. In the context of the plays, these tunes become the vehicle for newly composed lyrics, which are normally rhymed, and which normally follow the poetic scansion and structure of the original lyrics.  Rhyme, versification and structure are thus of interest to us."""

            act: int = Field(description="The act number in which this musical moment takes place. Will be labeled at the top of the act or scene in which it takes place.")
            scene: int = Field(description="The scene number in which the musical moment takes place. Will be labeled at the top of the scene.")
            number: int = Field(description = "The index of the musical moment in the scene. For example, if this is the first musical moment in the scene, this should be 1.")
            characters: list[str] = Field(description="the character or characters who are singing (or otherwise making music) within this specific musical moment,")
            dramatic_situation: str = Field(description="the dramatic situation (a love scene, a crowd scene) in which the musical moment is occurring")
            air_or_melodie: str = Field(description="The title of the 'air' or 'melodie' of which the musical moment is based. It will be labeled in the text as 'air' or 'melodie'.")
            poetic_text: str = Field(description="The text from the music number. Do not include stage directions, only the lyrics sung by the characters in this musical moment")
            rhyme_scheme: str = Field(description = "The rhyme scheme for the poetic text in the musical moment. For example, sentences that end in 'tree' 'be' 'why' and 'high' would have a rhyme scheme of AABB.")
            poetic_form: str = Field(description="form of the poetic text, which might involve some refrain")
            end_of_line_accents: list[str] = Field(description = "the end accent for each line (masculine or féminine)")
            syllable_count_per_line: list[int] = Field(description = "the number of syllables per line. look out for contractions and colloquialisms.that might make the count of syllables less than obvious. Normally a word like ‘voilà’ would of course have 2 syllables. But the musical rhythm of a particular melodie might require that it be _sung_ as only one syllable, as would be the case if the text reads ‘v’la’. Similarly ‘mademoiselle’ would have 4 syllables in spoken French. But the musical context might make it sound like 5. Or a character speaking dialect might sing “Mam’zelle”, which would have only 2 (or perhaps 3) syllables.")
            irregularities: Optional[str] = Field(description="any irregularities within the musical number")
            stage_direction_or_cues: Optional[str] = Field(description="any stage directions, which tell a character what to do, but aren't a part of another character's dialogue. These are usually connected with a character’s name, and often are in some contrasting typography (italics, or in parentheses - though this may not be picked up by the filereader).  Sometimes these directions even happen in the midst of a song! In a related way there are sometimes ‘cues’ for music, or performance (as when there is an offstage sound effect, or someone is humming) Most times the stage directions appear just before or after the song text. But sometimes they appear in the midst of the texts. The directions should be reported here and not in the transcription of the poem.")
            reprise: Optional[str] = Field(description="there are sometimes directions that indicate the ‘reprise’ of some earlier number or chorus.")

        class VaudevillePlay(BaseModel):
            musicalMoments: list[MusicalMoment] = Field(description="""A list of musical moments in a Vaudeville play, as MusicalMoment objects. Many of these musical moments reuse some preexisting (and often well-known)  melody or tune.  These are variously called "melodie”, or “air”, and identified with a short title that refers in some way to an opera or collection of melodies from which it was drawn.  The titles might include the names of works, or other characters in those original works. In the context of the plays, these tunes become the vehicle for newly composed lyrics, which are normally rhymed, and which normally follow the poetic scansion and structure of the original lyrics.  Rhyme, versification and structure are thus of interest to us.""")    

        structured_llm = llm.with_structured_output(VaudevillePlay)
        system_prompt = """
        You are a literary analyst specializing in French Vaudeville plays from the 18th century. 
        Your goal is to identify each musical moment in the text, and for each, extract detailed structured information, 
        including act, scene, characters, dramatic situation, air or melodie, poetic text, rhyme scheme, poetic form, end-of-line accents, syllable count, and any irregularities. 
        Some parts of the text were slightly misinterpreted by the file reader (e.g., missing spaces or strange line breaks).
        """
        human_prompt = """
        Given the following chunk of the play, analyze and return the musical moments as a structured VaudevillePlay object.
        """


        from typing_extensions import List, TypedDict, Optional
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            ("human","Context:\n{context}\n\nQuestion:\n{question}")
            ])

        class State(TypedDict):
            index: int
            context: Document
            answer: str

        def check_index(state: State):
            return state

        def retrieve_doc(state: State):
            document = all_splits[state["index"]]
            return {"context": document}

        def generate(state: State):
            i = state["index"]
            message = prompt.invoke({"question":human_prompt,"context" : f'Act {all_indexes.all_scenes[i].act}, Scene {all_indexes.all_scenes[i].scene}:\n\n {state["context"].page_content}'})
            response = structured_llm.invoke(message)
            return {"answer": response}

        from langgraph.graph import START, StateGraph

        graph_builder = StateGraph(State).add_sequence([check_index, retrieve_doc, generate])
        graph_builder.add_edge(START, "check_index")
        graph = graph_builder.compile()

        ###############################
        # Analyze Scenes and Export CSV
        ###############################

        def analyze_scenes(docs: List[Document]) -> List[MusicalMoment]:
            all_moments: List[MusicalMoment] = []

            for i,doc in enumerate(docs):
                response = graph.invoke({"index": i})
                moments = response["answer"].musicalMoments
                all_moments.extend(moments)
            
            return all_moments

        all_moments = analyze_scenes(all_splits)
        import csv

        # Convert all MusicalMoment objects to dicts
        moments_dicts = [moment.model_dump() for moment in all_moments]

        # Get all field names from the first moment
        fieldnames = moments_dicts[0].keys() if moments_dicts else []

        # Write to CSV
        # Convert all MusicalMoment objects to dicts
        moments_dicts = [moment.model_dump() for moment in all_moments]

        # Prepare in-memory CSV string
        output = io.StringIO()
        fieldnames = moments_dicts[0].keys() if moments_dicts else []
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in moments_dicts:
            for key, value in row.items():
                if isinstance(value, list):
                    row[key] = "; ".join(str(v) for v in value)
            writer.writerow(row)

        # Convert StringIO to bytes for download
        csv_bytes = output.getvalue().encode("utf-8")

        st.write("Download your CSV here:")
        # Display download button
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name="musical_moments.csv",
            mime="text/csv",
            use_container_width = True
        )

        

        df = pd.DataFrame(moments_dicts)
        st.markdown("### Musical Moments Preview")
        st.dataframe(df, use_container_width=True)
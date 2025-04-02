import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_data(show_spinner=False)
def get_cached_response(prompt, model):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful data assistant who only outputs code using pandas."},
            {"role": "user", "content": prompt}
        ]
    )

class DataFile:
    def __init__(self, name, file):
        self.name = name
        self.file = file
        self.df = self.load_file()

    def load_file(self):
        try:
            ext = self.name.split('.')[-1]
            if ext == "csv":
                return pd.read_csv(self.file)
            else:
                return pd.read_excel(self.file)
        except Exception as e:
            st.error(f"Failed to load {self.name}: {e}")
            return pd.DataFrame()

    def preview(self, n):
        st.subheader(f"Preview: {self.name} (Top {n} rows)")
        st.dataframe(self.df.head(n))

    def ask_question(self, question):
        st.subheader("GPT Answer")
        prompt = f"""You are a data assistant working with a pandas dataframe called `df`. Here's a preview of the data:

{self.df.head(15).to_csv(index=False)}

The user asked: "{question}"

Respond ONLY with Python code using `df`, and end with a variable `output` that I can display.
Do NOT limit the number of rows in the output. Do NOT use `.head()`, `.iloc`, or any row slicing.
Return the full filtered DataFrame as `output`. Do not include markdown or comments — just raw Python code.
"""


        response = get_cached_response(prompt, "gpt-3.5-turbo")

        code = response.choices[0].message.content
        code = code.strip("`")                  # Remove all backticks
        code = code.replace("python", "").strip()  # Remove "python" language tag
        st.code(code, language="python")

        try:
            local_scope = {"df": self.df}
            exec(code, {}, local_scope)
            output = local_scope.get("output")
            if isinstance(output, pd.DataFrame):
                st.subheader("Result")
                num_rows_to_show = st.number_input(
                "Number of rows to display from the result",
                min_value=1,
                max_value=len(output),
                value=min(5, len(output)),
                key=f"rows_{self.name}_{question}"
                )
                st.dataframe(output.head(num_rows_to_show))
            else:
                st.write(output)
        except Exception as e:
            st.error(f"Error running code: {e}")


class MultiFileApp:
    def __init__(self):
        self.files = {}

    def run(self):
        st.title("AI application")

        if "prompt_history" not in st.session_state:
            st.session_state.prompt_history = []

        uploaded = st.file_uploader("Upload multiple CSV or Excel files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)

        if uploaded:
            for f in uploaded:
                data_file = DataFile(f.name, f)
                self.files[f.name] = data_file

            selected = st.sidebar.selectbox("Select a file", list(self.files.keys()))
            if "last_file" in st.session_state and st.session_state.last_file != selected:
                st.session_state.last_question = None
            st.session_state.last_file = selected

            data = self.files[selected]

            num = st.sidebar.number_input("Top N rows", min_value=1, max_value=len(data.df), value=5)
            data.preview(num)
            
            with st.expander("Prompt History", expanded=True):
                history_container = st.container()
                with history_container:
                    for i, item in enumerate(reversed(st.session_state.prompt_history)):
                        if item["file"] == selected:
                            if st.button(f"🗂 {item['question']}", key=f"history_{i}"):
                                data.ask_question(item["question"])

                # Apply scroll to the *parent* container
                st.markdown("""
                    <style>
                    [data-testid="stExpander"] div[data-testid="stVerticalBlock"] > div {
                        max-height: 250px;
                        overflow-y: auto;
                    }
                    </style>
                """, unsafe_allow_html=True)

            if st.button("Clear Prompt History for this file"):
                st.session_state.prompt_history = [
                    item for item in st.session_state.prompt_history if item["file"] != selected
                ]


            question = st.text_input("Ask a question about this file", key="question_input")
            if st.button("Submit Question"):
                q_clean = question.strip()
                already_asked = any(
                    q_clean == item["question"] and selected == item["file"]
                    for item in st.session_state.prompt_history
                )

                if q_clean and not already_asked:
                    st.session_state.prompt_history.append({
                        "file": selected,
                        "question": q_clean
                    })

                st.session_state.last_question = q_clean

            # If user clicked on a past history item or just submitted one
            if "last_question" in st.session_state:
                data.ask_question(st.session_state.last_question)


# Run the app
if __name__ == "__main__":
    app = MultiFileApp()
    app.run()

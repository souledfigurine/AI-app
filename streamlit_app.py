import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

Respond ONLY with Python code using `df`, and end with a variable `output` that I can display. Do not include markdown or formatting — just raw Python code.
"""


        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful data assistant who only outputs code using pandas."},
                {"role": "user", "content": prompt}
            ]
        )

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
                value=min(5, len(output))
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

        uploaded = st.file_uploader("Upload multiple CSV or Excel files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)

        if uploaded:
            for f in uploaded:
                data_file = DataFile(f.name, f)
                self.files[f.name] = data_file

            selected = st.sidebar.selectbox("Select a file", list(self.files.keys()))
            data = self.files[selected]

            num = st.sidebar.number_input("Top N rows", min_value=1, max_value=len(data.df), value=5)
            data.preview(num)

            question = st.text_input("Ask a question about this file")
            if question:
                data.ask_question(question)


# Run the app
if __name__ == "__main__":
    app = MultiFileApp()
    app.run()

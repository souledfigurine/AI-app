# 🧭 User Guide: AI Data Assistant

## 📌 What This App Does

This app lets you:

- 📂 Upload CSV or Excel files
- 🤖 Ask natural-language questions about the data
- 🧠 Get Python code and results generated by OpenAI’s GPT-3.5
- 📊 View and control how many rows to display
- 🗂 Reuse past questions or clear prompt history per file

---

##🚀 How to Use It

### 1. Upload Your File

- Click "Browse files" and select one or more .csv, .xls, or .xlsx files.
- Files are loaded into memory; nothing is stored or shared.

### 2. Preview Your Data

- Use the "Top N rows" slider on the left to control how many rows of each file to preview.
- Switch between files using the dropdown.

### 3. Ask a Question

- Enter a natural-language question like:
  - “Show me the top 10 customers by revenue”
  - “What's the average salary by department?”
- Click "💬 Submit Question" to get results.

### 4. Read the Output

- GPT will generate Python code using pandas.
- Use the slider to control how many rows of the result to display.

### 5. Reuse or Clear Prompts

- All questions are saved per file.
- Click 🗂 to rerun past prompts.
- Click 🧹 Clear Prompt History to reset for that file.

---

##💬 Example Questions

- Try asking things like:
  - "List the top 5 products by profit"
  - "Show average age grouped by gender"
  - "Find rows where score is above 90"

import os
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from dotenv import load_dotenv
from google import genai

# Load API Key
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def load_data(file):
    try:
        df = pd.read_csv(file.name)
        return df
    except Exception as e:
        return str(e)

def pandas_analysis(df, question):

    question = question.lower()

    if "rows" in question:
        return f"Total Rows : {df.shape[0]}"

    elif "columns" in question:
        return "\n".join(df.columns)

    elif "missing" in question:
        return str(df.isnull().sum())

    elif "summary" in question:
        return str(df.describe())

    elif "head" in question:
        return str(df.head())

    return None

# Gemini Analysis
def ask_gemini(df, question):

    sample = df.head(20).to_string()

    prompt = f"""
You are an expert AI Data Analyst.

Dataset Sample:

{sample}

Question:

{question}

Give a clear answer.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# Create Charts

def create_chart(df, question):

    question = question.lower()

    if "chart" in question or "plot" in question or "graph" in question:

        column = None

        for col in df.columns:
            if col.lower() in question:
                column = col
                break

        if column:
            plt.figure(figsize=(8,5))
            df[column].value_counts().head(10).plot(kind="bar")
            plt.title(column)

            chart_path = "chart.png"
            plt.savefig(chart_path)
            plt.close()

            return chart_path

    return None

# Main Chatbot

def chatbot(file, question):

    df = load_data(file)

    if isinstance(df, str):
        return df, None

    answer = pandas_analysis(df, question)

    if answer is None:
        answer = ask_gemini(df, question)

    chart = create_chart(df, question)

    return answer, chart

# Gradio Interface

demo = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.File(label="Upload CSV File"),
        gr.Textbox(
            label="Ask your question"
        )
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Image(label="Chart")
    ],
    title="AI Data Analyst Chatbot",
    description="Upload a CSV file and ask questions about your data."
)

demo.launch()
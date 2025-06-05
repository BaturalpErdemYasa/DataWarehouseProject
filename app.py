# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 14:33:02 2025

@author: Btrlp8
"""

from flask import Flask, render_template, request, send_file
import os
import fitz  # PyMuPDF
from transformers import pipeline
import tempfile

app = Flask(__name__)
summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer_t5 = pipeline("summarization", model="t5-base")

# PDF'ten metin çıkaran yardımcı fonksiyon
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text

# Uzun metni parçalara ayırarak özetleyen fonksiyon
def summarize_long_text(text, summarizer, chunk_size=1000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []

    for i, chunk in enumerate(chunks):
        try:
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            summaries.append(f"Parça {i+1} Özeti:\n{summary}")
        except Exception as e:
            summaries.append(f"Parça {i+1} özetlenemedi: {str(e)}")

    return "\n\n".join(summaries)

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Dosya yüklendiğinde özetleme
@app.route('/summarize', methods=['POST'])
def summarize():
    uploaded_file = request.files['pdf_file']
    model_choice = request.form.get('model_choice', 'bart')
    if uploaded_file and uploaded_file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(uploaded_file)
        if model_choice == 't5':
            summary = summarize_long_text(text, summarizer_t5)
        else:
            summary = summarize_long_text(text, summarizer_bart)

        # Geçici dosyaya yaz
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
        temp.write(summary)
        temp.close()
        return send_file(temp.name, as_attachment=True, download_name="ozet.txt")

    return "Geçersiz dosya formatı. Lütfen PDF yükleyin."

if __name__ == '__main__':
    app.run(debug=False)

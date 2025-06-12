# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 14:33:02 2025

@author: Btrlp8
"""

from flask import Flask, render_template, request, send_file
import os
import re
import fitz  # PyMuPDF
from transformers import pipeline
import tempfile
import nltk
from bertopic import BERTopic
from pyvis.network import Network
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
import torch  # En üste ekleyin

app = Flask(__name__)

# Summarization pipeline'ları
def get_summarizer_bart():
    if not hasattr(get_summarizer_bart, '_instance'):
        get_summarizer_bart._instance = pipeline(
            "summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1
        )
    return get_summarizer_bart._instance

def get_summarizer_t5():
    if not hasattr(get_summarizer_t5, '_instance'):
        get_summarizer_t5._instance = pipeline(
            "summarization", model="t5-base", device=0 if torch.cuda.is_available() else -1
        )
    return get_summarizer_t5._instance


def get_summarizer_t5small():
    if not hasattr(get_summarizer_t5small, '_instance'):
        get_summarizer_t5small._instance = pipeline(
            "summarization", model="t5-small", device=0 if torch.cuda.is_available() else -1
        )
    return get_summarizer_t5small._instance

def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def extract_sentences_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return sentences

def summarize_long_text(text, summarizer, chunk_size=2000, batch_size=32):
    """
    Daha hızlı özetleme için:
    - chunk_size artırıldı (2000 karakter)
    - batch processing eklendi
    """
    if not text.strip():
        return "Metin bulunamadı."
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []

    # Batch processing
    for i in range(0, len(chunks), batch_size):
        batch = [chunk for chunk in chunks[i:i+batch_size] if chunk.strip()]
        if not batch:
            continue
        try:
            results = summarizer(
                batch,
                max_length=120,  # daha kısa özet için
                min_length=20,
                do_sample=False
            )
            for j, res in enumerate(results):
                summaries.append(f"Parça {i+j+1} Özeti:\n{res['summary_text']}")
        except Exception as e:
            for j in range(len(batch)):
                summaries.append(f"Parça {i+j+1} özetlenemedi: {str(e)}")

    return "\n\n".join(summaries) if summaries else "Özet oluşturulamadı."

# Yeni: BERTopic parametreleri
vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=3000
)

umap_model = UMAP(
    n_neighbors=5,
    n_components=2,
    min_dist=0.0,
    metric='cosine'
)

hdbscan_model = HDBSCAN(
    min_cluster_size=2,
    min_samples=1,
    cluster_selection_method='eom'
)

embedding_model = SentenceTransformer("paraphrase-MiniLM-L12-v2")

# BERTopic modeli (önceden tanımlı)
topic_model = BERTopic(
    language="english",
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    min_topic_size=2,
    nr_topics=5,
    calculate_probabilities=False,
    verbose=True
)

def extract_topics_with_bertopic(sentences):
    try:
        if not sentences or len(sentences) < 5:
            raise ValueError("En az 5 cümle gerekli.")
        
        valid_sentences = [s for s in sentences if s.strip() and len(s.strip()) > 20]
        
        if len(valid_sentences) < 5:
            raise ValueError(f"Yeterli geçerli cümle bulunamadı. Bulunan: {len(valid_sentences)}, Gerekli: 5")
        
        print(f"Analiz edilecek cümle sayısı: {len(valid_sentences)}")
        
        topics, probabilities = topic_model.fit_transform(valid_sentences)
        
        unique_topics = set(topics)
        print(f"Bulunan konu sayısı: {len(unique_topics)}")
        
        if len(unique_topics) <= 1:
            raise ValueError("Anlamlı konu bulunamadı.")
        
        return topic_model, topics
    except Exception as e:
        raise Exception(f"Konu analizi hatası: {str(e)}")

def visualize_topics_with_pyvis(topic_model):
    try:
        net = Network(
            height="700px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#222222",
            notebook=False
        )
        topics_dict = topic_model.get_topics()
        valid_topics = {k: v for k, v in topics_dict.items() if k != -1}
        if not valid_topics:
            raise ValueError("Görselleştirilebilir konu bulunamadı.")
        
        for topic_num, topic_words in valid_topics.items():
            top_words = [word for word, score in topic_words[:3]]
            topic_label = " ".join(top_words).title()
            
            net.add_node(f"topic_{topic_num}", label=topic_label, color="#4e79a7", size=30)
            
            prev_word_node = None
            for i, (word, score) in enumerate(topic_words[:5]):
                word_node = f"word_{topic_num}_{i}_{word}"
                net.add_node(word_node, label=word, color="#a0cbe8", size=18)
                net.add_edge(f"topic_{topic_num}", word_node, width=2)
                if prev_word_node:
                    net.add_edge(prev_word_node, word_node, width=1, color="#b2b2b2")
                prev_word_node = word_node
        
        topic_nums = list(valid_topics.keys())
        for i in range(len(topic_nums)-1):
            net.add_edge(f"topic_{topic_nums[i]}", f"topic_{topic_nums[i+1]}", width=1, color="#f28e2b")
        
        html_path = os.path.join(tempfile.gettempdir(), f"bertopic_pyvis_{os.getpid()}.html")
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 200}
          }
        }
        """)
        net.show(html_path, notebook=False)
        if not os.path.exists(html_path):
            raise ValueError("HTML dosyası oluşturulamadı.")
        return html_path
    except Exception as e:
        raise Exception(f"Görselleştirme hatası: {str(e)}")

def create_simple_topic_html(topic_model):
    try:
        topics_dict = topic_model.get_topics()
        valid_topics = {k: v for k, v in topics_dict.items() if k != -1}
        
        if not valid_topics:
            raise ValueError("Görselleştirilebilir konu bulunamadı.")
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Konu Analizi Sonuçları</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .topic { margin: 20px 0; padding: 15px; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .topic h3 { color: #333; margin-top: 0; }
                .words { display: flex; flex-wrap: wrap; gap: 10px; }
                .word { background-color: #4ecdc4; color: white; padding: 5px 10px; border-radius: 15px; font-size: 14px; }
                .score { font-size: 12px; opacity: 0.8; }
            </style>
        </head>
        <body>
            <h1>Konu Analizi Sonuçları</h1>
        """
        
        for topic_num, words in valid_topics.items():
            html_content += f'<div class="topic"><h3>Konu {topic_num}</h3><div class="words">'
            for word, score in words[:5]:
                html_content += f'<span class="word">{word} <span class="score">({score:.3f})</span></span>'
            html_content += '</div></div>'
        
        html_content += """
        </body>
        </html>
        """
        
        html_path = os.path.join(tempfile.gettempdir(), f"topic_analysis_{os.getpid()}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    except Exception as e:
        raise Exception(f"Basit HTML oluşturma hatası: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        uploaded_file = request.files.get('pdf_file')
        if not uploaded_file or not uploaded_file.filename:
            return "Dosya seçilmedi.", 400
        
        if not uploaded_file.filename.lower().endswith('.pdf'):
            return "Geçersiz dosya formatı. Lütfen PDF yükleyin.", 400
        
        model_choice = request.form.get('model_choice', 'bart')
        file_bytes = uploaded_file.read()
        text = extract_text_from_pdf(file_bytes)
        
        if not text.strip():
            return "PDF'den metin çıkarılamadı.", 400
        
        if model_choice == 't5':
            summarizer = get_summarizer_t5()
        elif model_choice == 'distilbart':
            summarizer = get_summarizer_distilbart()
        elif model_choice == 't5small':
            summarizer = get_summarizer_t5small()
        else:
            summarizer = get_summarizer_bart()
        
        summary = summarize_long_text(text, summarizer)
        
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
        temp.write(summary)
        temp.close()
        
        return send_file(temp.name, as_attachment=True, download_name="ozet.txt")
    
    except Exception as e:
        return f"Hata oluştu: {str(e)}", 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

@app.route('/topics', methods=['POST'])
def topics():
    try:
        uploaded_file = request.files.get('pdf_file')
        if not uploaded_file or not uploaded_file.filename:
            return "Dosya seçilmedi.", 400
        
        if not allowed_file(uploaded_file.filename):
            return "Geçersiz dosya formatı. Lütfen PDF yükleyin.", 400
        
        file_bytes = uploaded_file.read()
        sentences = extract_sentences_from_pdf(file_bytes)
        
        if not sentences:
            return "PDF'den cümle çıkarılamadı.", 400
        
        print(f"Toplam cümle sayısı: {len(sentences)}")
        
        topic_model_result, topics = extract_topics_with_bertopic(sentences)
        
        try:
            html_path = visualize_topics_with_pyvis(topic_model_result)
        except Exception as pyvis_error:
            print(f"Pyvis hatası: {pyvis_error}")
            html_path = create_simple_topic_html(topic_model_result)
        
        return send_file(html_path, as_attachment=True, download_name="konu_gorseli.html")
    
    except Exception as e:
        print(f"Genel hata: {str(e)}")
        return f"Konu analizi hatası: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=False)

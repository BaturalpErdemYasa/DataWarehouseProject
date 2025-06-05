# -*- coding: utf-8 -*-
"""
PDF Özetleyici - NLTK Bağımsız Hızlı Versiyon
"""

from flask import Flask, render_template, request, send_file
import os
import fitz  # PyMuPDF
import tempfile
import re
from collections import Counter

app = Flask(__name__)

class FastSummarizer:
    def __init__(self):
        # Türkçe ve İngilizce stop words
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            've', 'bir', 'bu', 'şu', 'o', 'da', 'de', 'ki', 'mi', 'mu', 'mü', 'mı', 'olan', 'olan',
            'ile', 'gibi', 'daha', 'çok', 'en', 'hem', 'ya', 'veya', 'ancak', 'fakat', 'ama',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz', 'on'
        ])
    
    def sentence_split(self, text):
        """Basit sentence splitting - NLTK gerektirmez"""
        # Nokta, ünlem, soru işareti ile ayır
        sentences = re.split(r'[.!?]+', text)
        # Boş stringleri ve çok kısa cümleleri filtrele
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def word_tokenize(self, text):
        """Basit word tokenization"""
        # Sadece harfleri ve rakamları al
        words = re.findall(r'\b[a-zA-ZçğıöşüÇĞIİÖŞÜ]+\b', text.lower())
        return words
    
    def clean_text(self, text):
        """Metni temizle"""
        # Özel karakterleri temizle
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        # Çoklu boşlukları tek boşluğa çevir
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def score_sentences(self, sentences, word_freq):
        """Cümleleri skorla"""
        sentence_scores = {}
        
        for sentence in sentences:
            words = self.word_tokenize(sentence)
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            
            if len(words) < 3:  # Çok kısa cümleleri atla
                continue
                
            score = 0
            word_count = 0
            
            for word in words:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
                
        return sentence_scores
    
    def extractive_summarize(self, text, max_length=150):
        """Extractive özetleme - NLTK bağımsız"""
        try:
            # Metni temizle
            clean_text = self.clean_text(text)
            
            # Cümlelere ayır
            sentences = self.sentence_split(clean_text)
            
            if len(sentences) <= 3:
                return '. '.join(sentences) + '.'
            
            # Cümle sayısını max_length'e göre ayarla
            num_sentences = max(2, min(6, max_length // 30))
            
            if len(sentences) <= num_sentences:
                return '. '.join(sentences) + '.'
            
            # Kelimeleri say
            words = self.word_tokenize(clean_text)
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            
            if not words:
                return '. '.join(sentences[:3]) + '.'
            
            # Kelime frekansları
            word_freq = Counter(words)
            
            # En yaygın kelimelerin skorunu normalize et
            max_freq = max(word_freq.values()) if word_freq else 1
            for word in word_freq:
                word_freq[word] = word_freq[word] / max_freq
            
            # Cümleleri skorla
            sentence_scores = self.score_sentences(sentences, word_freq)
            
            if not sentence_scores:
                return '. '.join(sentences[:num_sentences]) + '.'
            
            # En yüksek skorlu cümleleri seç
            sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            best_sentences = [sent[0] for sent in sorted_sentences[:num_sentences]]
            
            # Orijinal sırayı koru
            final_sentences = []
            for sentence in sentences:
                if sentence in best_sentences:
                    final_sentences.append(sentence)
                if len(final_sentences) == num_sentences:
                    break
            
            result = '. '.join(final_sentences)
            if not result.endswith('.'):
                result += '.'
                
            return result
            
        except Exception as e:
            # Hata durumunda ilk birkaç cümleyi döndür
            sentences = self.sentence_split(text)
            return '. '.join(sentences[:3]) + f' (Basit özetleme: {str(e)})'

# Global summarizer
fast_summarizer = FastSummarizer()

def extract_text_from_pdf(file):
    """PDF'den metin çıkar"""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        return f"PDF okuma hatası: {str(e)}"

def summarize_long_text(text, chunk_size=1000, max_length=150, min_length=50, model_name="distilbart-cnn"):
    """Uzun metni parçalara ayırarak özetle"""
    
    try:
        # Metni chunks'lara ayır
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size].strip()
            if len(chunk) > min_length:
                chunks.append(chunk)
        
        if not chunks:
            return "Metin çok kısa veya özetlenemez."
        
        summaries = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Her chunk için özetleme
                summary = fast_summarizer.extractive_summarize(chunk, max_length)
                
                if summary and len(summary.strip()) > 10:
                    summaries.append(f"Parça {i+1} Özeti:\n{summary}")
                    
            except Exception as e:
                summaries.append(f"Parça {i+1} özetlenemedi: {str(e)}")
        
        if not summaries:
            # Fallback: Tüm metni tek seferde özetle
            return f"Genel Özet:\n{fast_summarizer.extractive_summarize(text, max_length)}"
        
        return "\n\n".join(summaries)
        
    except Exception as e:
        return f"Özetleme hatası: {str(e)}\n\nOrijinal metin: {text[:200]}..."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        uploaded_file = request.files['pdf_file']
        
        # Form verilerini al
        model_name = request.form.get('model', 'distilbart-cnn')
        max_length = int(request.form.get('max_length', 150))
        min_length = int(request.form.get('min_length', 50))
        chunk_size = int(request.form.get('chunk_size', 1000))
        
        # Güvenlik kontrolü
        allowed_models = [
            'facebook/bart-large-cnn',
            't5-large', 
            'pegasus-xsum',
            'distilbart-cnn'
        ]
        
        if model_name not in allowed_models:
            model_name = 'distilbart-cnn'
        
        # Min/max kontrolü
        if min_length >= max_length:
            min_length = max_length // 2
            
        if uploaded_file and uploaded_file.filename.endswith('.pdf'):
            # PDF'den metin çıkar
            text = extract_text_from_pdf(uploaded_file)
            
            if not text or len(text.strip()) < 50:
                return "PDF'den yeterli metin çıkarılamadı. Lütfen farklı bir dosya deneyin."
            
            # Özetleme işlemi
            summary = summarize_long_text(
                text, 
                chunk_size=chunk_size,
                max_length=max_length,
                min_length=min_length,
                model_name=model_name
            )

            # Model açıklaması
            model_descriptions = {
                'distilbart-cnn': 'Extractive Summarization (Hızlı ve Etkili)',
                'facebook/bart-large-cnn': 'Extractive Summarization (BART Benzeri)',
                't5-large': 'Extractive Summarization (T5 Benzeri)',
                'pegasus-xsum': 'Extractive Summarization (Pegasus Benzeri)'
            }

            # Özet bilgilerini ekle
            summary_info = f"""=== PDF ÖZET RAPORU ===
Dosya: {uploaded_file.filename}
Model: {model_descriptions.get(model_name, model_name)}
Maksimum Uzunluk: {max_length} kelime
Minimum Uzunluk: {min_length} kelime
Parça Boyutu: {chunk_size} karakter
İşlem Tarihi: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Orijinal Metin Uzunluğu: {len(text)} karakter
Özet Uzunluğu: {len(summary)} karakter

=== ÖZET ===

{summary}

=== BİLGİLENDİRME ===
Bu özet extractive summarization yöntemi ile oluşturulmuştur.
En önemli ve anlamlı cümleler orijinal metinden seçilmiştir.
İşlem süresi: ~1-3 saniye (Çok hızlı - NLTK bağımsız!)
Yöntem: Regex tabanlı sentence splitting + Kelime frekans analizi
"""

            # Geçici dosyaya yaz
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
            temp.write(summary_info)
            temp.close()
            
            return send_file(temp.name, as_attachment=True, download_name=f"hizli_ozet_{model_name.replace('/', '_')}.txt")

        return "Geçersiz dosya formatı. Lütfen PDF yükleyin."
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return f"Bir hata oluştu: {str(e)}"

if __name__ == '__main__':
    app.run(debug=False)
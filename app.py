from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import pdfplumber
import spacy
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from sentence_transformers import SentenceTransformer, util
import re
import os
import time

app = Flask(__name__)

# Load models
nlp = spacy.load('en_core_web_sm')
nltk.download('punkt')
qg_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
qg_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
qa_pipeline = pipeline('question-answering', model='deepset/roberta-large-squad2')
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Text extraction function
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += ' '.join(page_text.split()) + "\n"
    return text

# Question generation
def generate_questions(text, num_questions):
    sentences = [sent.text for sent in nlp(text).sents]
    questions = []
    for sentence in sentences:
        if len(questions) >= num_questions:
            break
        input_text = f"generate question: {sentence}"
        input_ids = qg_tokenizer.encode(input_text, return_tensors="pt")
        output_ids = qg_model.generate(input_ids, max_length=64, num_beams=2, top_k=20, top_p=0.9, temperature=0.7, do_sample=True, num_return_sequences=1)
        for output in output_ids:
            question = qg_tokenizer.decode(output, skip_special_tokens=True)
            if question and question not in questions:
                questions.append(question)
    return questions

# Answer-finding
def find_answers(questions, context):
    qa_pairs = []
    for question in questions:
        result = qa_pipeline(question=question, context=context)
        if result['score'] > 0.7:
            qa_pairs.append({'question': question, 'answer': result['answer']})
    return qa_pairs

# Word count function
def count_occurrences(word, text):
    word = word.lower()
    text = text.lower()
    separate_word_count = len(re.findall(r'\b' + re.escape(word) + r'\b', text))
    substring_count = text.count(word)
    return separate_word_count, substring_count

# Route to handle file upload and question generation
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    num_questions = int(request.form.get('num_questions', 5))
    file.save(file.filename)
    text = extract_text_from_pdf(file.filename)
    os.remove(file.filename)
    questions = generate_questions(text, num_questions)
    qa_pairs = find_answers(questions, text)
    return jsonify(qa_pairs)

# Route to handle word count
@app.route('/count_word', methods=['POST'])
def count_word():
    if 'file' not in request.files or 'word' not in request.form:
        return jsonify({'error': 'File or word not provided'}), 400
    file = request.files['file']
    word = request.form['word']
    file.save(file.filename)
    text = extract_text_from_pdf(file.filename)
    os.remove(file.filename)
    separate_count, substring_count = count_occurrences(word, text)
    return jsonify({
        'word': word,
        'separate_word_count': separate_count,
        'substring_count': substring_count
    })

if __name__ == '__main__':
    app.run(debug=True)

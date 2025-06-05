from flask import Flask, render_template, request, session, redirect, send_from_directory
import os
from werkzeug.utils import secure_filename
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import nltk
from nltk.tokenize import sent_tokenize
from random import shuffle
from pptx import Presentation
from keybert import KeyBERT
from pdfminer.high_level import extract_text
from pathlib import Path
import shutil
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the model and tokenizer directly
explainer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
kw_model = KeyBERT()


app = Flask(__name__)
app.secret_key = 'Notes2QA'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def extract_text_from_pptx(path):
    prs = Presentation(path)
    full_text = []
    for slide in prs.slides:
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text.append(shape.text.strip())
        # Join slide content as paragraph
        if slide_text:
            full_text.append(" ".join(slide_text))
    return "\n".join(full_text)


def extract_pdf_text(file_path):
    text = extract_text(file_path)
    return text


def is_valid_question(q):
    question_words = ('what', 'why', 'how', 'when', 'which', 'who', 'is', 'are', 'does', 'can')
    return q.endswith('?') and len(q.split()) > 5 and "?" in q and q.lower().startswith(question_words)


@app.route('/', methods=['GET', 'POST'])
def index():
    questions = []
    filename = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            session['file_path'] = file_path
            print(session['file_path'])
        else:
            file_path = session.get('file_path')

        filename = os.path.basename(file_path)
        path = Path(f"explanations/{filename}")
        if path.exists():
            print("File exists.")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                chunks = content.strip().split("\n\n")
                print(f"Loaded Chunks: \n{chunks}\n")
        else:
            print("File does not exist.")
            data = ""

            if file_path.lower().endswith(".pptx"):
                data = extract_text_from_pptx(file_path)

            elif file_path.lower().endswith(".docx"):
                from docx import Document
                doc = Document(file_path)
                for para in doc.paragraphs:
                    data += para.text + "\n"

            elif file_path.lower().endswith(".pdf"):
                data = extract_pdf_text(file_path)

            else:  # default to .txt
                with open(file_path, "r", encoding="utf-8") as f:
                    data = f.read()

            sentences = sent_tokenize(data)
            num_sentences = len(sentences)
            print(f"{num_sentences} sentences")

            # Dynamically scale size (number of sentences per chunk)
            if num_sentences < 10:
                size = 1
            elif num_sentences < 50:
                size = 2
            elif num_sentences < 100:
                size = 4
            else:
                size = 6  # More aggressive chunking for large files

            # --- Create Chunks ---
            print(f"\nCreating chunks....\n")
            chunks = []
            for i in range(0, len(sentences), size):
                chunk = " ".join(sentences[i:i + size])
                chunk = explainer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                print(f"Explanation {i / size}:\n{chunk}")
                chunks.append(chunk.strip())

            with open(path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(chunk + "\n\n")  # two newlines for spacing

        shuffle(chunks)  # Randomize order to avoid bias from the top of file

        # --- Generate Questions ---
        questions = []
        used_chunks = set()

        print("Generating questions...")
        for chunk in chunks:
            if len(questions) >= 10:
                break
            if chunk in used_chunks:
                continue

            input_text = f"Generate a question: {chunk}"
            result = generator(input_text, do_sample=False)
            q_text = result[0]['generated_text'].replace("question:", "").strip()

            if is_valid_question(q_text):
                questions.append(q_text)
                used_chunks.add(chunk)
                print(f"Q: {q_text}")

            path = Path(f"questions/questions.txt")
            with open(path, "w", encoding="utf-8") as f:
                for q in questions:
                    f.write(q + "\n")

    return render_template('index.html', questions=questions, filename=filename)


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    try:
        # Example: if you store chunks in 'cache/chunks.txt'
        cache_path = 'explanations/'  # Or a folder like 'cache/'

        if os.path.exists(cache_path):
            for filename in os.listdir(cache_path):
                file_path = os.path.join(cache_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        # Optional: delete subdirectories too
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

        return redirect('/')  # Redirect back to home
    except Exception as e:
        return f"Failed to clear cache: {e}", 500


@app.route('/download')
def download_questions():
    questions_dir = os.path.join(os.getcwd(), "questions")
    questions_file = os.path.join(questions_dir, "questions.txt")

    if os.path.exists(questions_file):
        return send_from_directory(directory=questions_dir, path="questions.txt", as_attachment=True)
    else:
        return "Questions file not found", 404


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

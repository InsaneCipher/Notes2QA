from flask import Flask, render_template, request, session
import os
from werkzeug.utils import secure_filename
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# Load the model and tokenizer directly
model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Define pipeline
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


app = Flask(__name__)
app.secret_key = 'Notes2QA'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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
            session['filename'] = file_path
        else:
            filename = session.get('filename')

        f = open(f"uploads/{filename}")
        data = f.read()
        sentences = sent_tokenize(data)

        questions = []
        for sent in sentences:
            input_text = f"generate question: {sent}"
            result = generator(input_text, do_sample=False)
            q_text = result[0]['generated_text'].replace("question:", "").strip()
            questions.append(q_text)

    return render_template('index.html', questions=questions, filename=filename)


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

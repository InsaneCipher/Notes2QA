from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from transformers import pipeline

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    questions = []
    filename = None

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file and uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)

            # Placeholder questions — replace with AI logic later
            questions = [
                "What is the main topic of the uploaded material?",
                "Summarize the key points in one paragraph.",
                "What are the important dates or definitions mentioned?"
            ]

            text = "Hi"

            generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-prepend")
            question = generator(text)

    return render_template('index.html', questions=questions, filename=filename)


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

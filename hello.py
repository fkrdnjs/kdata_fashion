from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file:
                style_info = process_image(image_file)
                return jsonify(style_info)
    return render_template('index.html')

def process_image(image_file):
    # 여기에 이미지 분석 파일 넣으면 될듯 싶어용

    return {'style': 'vintage', 'shop': '뭐있지'}

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/content')
def content():
    return render_template('content.html')

@app.route('/select')
def select():
    return render_template('select.html')
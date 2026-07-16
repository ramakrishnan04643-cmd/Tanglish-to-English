from flask import Flask, render_template, request, jsonify
from translator import translate_tanglish

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = translate_tanglish(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 so it is accessible from your phone
    app.run(host='0.0.0.0', port=5000, debug=True)
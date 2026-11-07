"""Simple web interface (Flask) for the spam classifier.

Run from the project root:   python src/app.py   then open http://127.0.0.1:5000
(Flask is used because it is lightweight; Streamlit or Tkinter would work too.)
"""
from flask import Flask, render_template_string, request

from predict import predict_message

app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spam Message Classifier</title>
<style>
  body{font-family:system-ui,sans-serif;background:#f1f3f7;margin:0;padding:24px;color:#1e2733}
  main{max-width:640px;margin:auto;background:#fff;border-radius:14px;padding:28px;box-shadow:0 2px 12px #0001}
  h1{margin-top:0;font-size:1.5rem} textarea{width:100%;height:120px;padding:10px;font:inherit;border:1px solid #c5ccd6;border-radius:8px;box-sizing:border-box}
  button{margin-top:12px;background:#4c7a6e;color:#fff;border:0;padding:10px 20px;border-radius:8px;font-size:1rem;cursor:pointer}
  .res{margin-top:20px;padding:14px;border-radius:8px}
  .spam{background:#fde8e8;border:1px solid #e45756}.ham{background:#e6f4ea;border:1px solid #54a24b}.warn{background:#fff6dd;border:1px solid #e0b000}
  small{color:#6b7684}
</style></head><body><main>
<h1>Spam Message Classifier</h1>
<p><small>Type or paste an SMS / email text and check whether it looks like spam.</small></p>
<form method="post"><textarea name="message" placeholder="e.g. You won a free prize! Click now to claim.">{{ message }}</textarea>
<br><button type="submit">Check message</button></form>
{% if result %}
  {% if not result.ok %}<div class="res warn">{{ result.message }}</div>
  {% else %}<div class="res {{ result.label }}"><strong>{{ "SPAM" if result.label == "spam" else "NOT SPAM (ham)" }}</strong><br>
  Spam probability: {{ "%.0f"|format(result.spam_probability*100) }}%</div>{% endif %}
{% endif %}
</main></body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    message, result = "", None
    if request.method == "POST":
        message = request.form.get("message", "")
        result = predict_message(message)
    return render_template_string(PAGE, message=message, result=result)


if __name__ == "__main__":
    app.run(debug=False)

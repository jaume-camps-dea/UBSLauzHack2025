from flask import Flask, render_template, request, jsonify
from model.model import pipeline
from google.cloud import bigquery, bigquery_storage
from google.oauth2 import service_account

app = Flask(__name__)

# Upload the key file to your notebook environment and use it
credentials = service_account.Credentials.from_service_account_file(
    'service_account/n8n-automation-g-474119-22912650c839.json'
)
client = bigquery.Client(credentials=credentials, project='n8n-automation-g-474119')

@app.route("/")
def index():
    return render_template("user_chatbot.html")

@app.route("/daily")
def daily():
    return render_template("ubs.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        result = pipeline(user_input,client)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

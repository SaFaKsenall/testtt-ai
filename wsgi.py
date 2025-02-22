from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'GreyMatters'

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Koyeb'in verdiği PORT'u kullan
    app.run(host="0.0.0.0", port=port)   # Tüm bağlantılara açık çalıştır

from flask import Flask
from bot import app  # Telegram botu için gerekli olan app'i buradan alıyoruz

# Flask uygulamasını oluştur
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Bot is running!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)  # Flask uygulamasını başlat

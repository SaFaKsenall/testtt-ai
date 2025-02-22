#!/bin/bash
gunicorn --bind 0.0.0.0:5000 wsgi:app &  # Flask arka planda çalışır
python main.py  # Telegram botu çalışır ve açık kalır
wait  # Arka planda çalışan işlemleri beklet

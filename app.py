import os
import pickle
import json
from datetime import datetime
from collections import Counter

from flask import Flask, request, jsonify, render_template

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import predict_with_confidence
from dataset import CATEGORY_LABELS

# ИНИЦИАЛИЗАЦИЯ FLASK-ПРИЛОЖЕНИЯ
app = Flask(__name__)

# ЗАГРУЗКА МОДЕЛИ И ТОКЕНАЙЗЕРА
SAVE_DIR = "saved_model"

print("Загрузка модели...")
model     = tf.keras.models.load_model(os.path.join(SAVE_DIR, "classifier.keras"))
tokenizer = pickle.load(open(os.path.join(SAVE_DIR, "tokenizer.pkl"), "rb"))
params    = pickle.load(open(os.path.join(SAVE_DIR, "params.pkl"), "rb"))

MAX_LEN               = params["max_len"]
UNCERTAINTY_THRESHOLD = params["uncertainty_threshold"]

print(f"Модель загружена. MAX_LEN={MAX_LEN}, порог={UNCERTAINTY_THRESHOLD}")

# СТАТИСТИКА
stats = {
    "total_requests":    0,
    "redirected_to_human": 0,
    "category_counts":   Counter(),
    "start_time":        datetime.now().isoformat(),
}

# Иконки и описания категорий для интерфейса
CATEGORY_META = {
    0: {"icon": "🔧", "color": "#4C72B0", "description": "Технические проблемы"},
    1: {"icon": "💳", "color": "#DD8452", "description": "Вопросы оплаты и счетов"},
    2: {"icon": "👤", "color": "#55A868", "description": "Управление аккаунтом"},
    3: {"icon": "💡", "color": "#8172B2", "description": "Запрос новой функции"},
    4: {"icon": "⚠️", "color": "#C44E52", "description": "Жалоба или претензия"},
    5: {"icon": "🔒", "color": "#2E86AB", "description": "Безопасность и конфиденциальность"},
    6: {"icon": "📖", "color": "#937860", "description": "Консультация по использованию"},
}

# МАРШРУТЫ
@app.route("/")
def index():
    return render_template("index.html", categories=CATEGORY_META)


@app.route("/classify", methods=["POST"])
def classify():
    # request.get_json() — парсит тело запроса как JSON
    data = request.get_json()

    if not data or "text" not in data:
        # jsonify преобразует словарь Python в JSON-ответ
        return jsonify({"error": "Поле 'text' обязательно"}), 400

    text = str(data["text"]).strip()

    if len(text) < 3:
        return jsonify({"error": "Текст слишком короткий"}), 400

    if len(text) > 1000:
        return jsonify({"error": "Текст слишком длинный (макс. 1000 символов)"}), 400

    # Предобработка текста
    # Тот же пайплайн, что и при обучении
    sequence = tokenizer.texts_to_sequences([text])
    padded   = pad_sequences(sequence, maxlen=MAX_LEN, padding="pre")

    # Предсказания
    results = predict_with_confidence(model, padded, uncertainty_threshold=UNCERTAINTY_THRESHOLD)
    result  = results[0]

    # Обновление статистики
    stats["total_requests"] += 1
    if result["needs_human"]:
        stats["redirected_to_human"] += 1
    else:
        stats["category_counts"][result["predicted_class"]] += 1

    # Формирование ответа
    pred_class = result["predicted_class"]
    meta = CATEGORY_META[pred_class]

    # Вероятности по всем категориям — для визуализации
    probs_detail = []
    for cat_id, prob in enumerate(result["all_probs"]):
        probs_detail.append({
            "category_id":   cat_id,
            "category_name": CATEGORY_LABELS[cat_id],
            "icon":          CATEGORY_META[cat_id]["icon"],
            "color":         CATEGORY_META[cat_id]["color"],
            "probability":   prob,
            "percentage":    round(prob * 100, 1),
        })
    # Сортировка по убыванию вероятности
    probs_detail.sort(key=lambda x: x["probability"], reverse=True)

    response = {
        "text":              text,
        "predicted_class":   pred_class,
        "category_name":     meta["description"],
        "icon":              meta["icon"],
        "color":             meta["color"],
        "confidence":        result["confidence"],
        "confidence_pct":    round(result["confidence"] * 100, 1),
        "needs_human":       result["needs_human"],
        "all_probabilities": probs_detail,
        "timestamp":         datetime.now().strftime("%H:%M:%S"),
    }

    return jsonify(response)


@app.route("/api/stats")
def get_stats():

    human_pct = 0
    if stats["total_requests"] > 0:
        human_pct = round(stats["redirected_to_human"] / stats["total_requests"] * 100, 1)

    return jsonify({
        "total_requests":       stats["total_requests"],
        "redirected_to_human":  stats["redirected_to_human"],
        "human_redirect_pct":   human_pct,
        "category_counts":      dict(stats["category_counts"]),
        "start_time":           stats["start_time"],
        "model_threshold":      UNCERTAINTY_THRESHOLD,
    })


@app.route("/api/examples")
def get_examples():
    examples = {
        "0": "Приложение вылетает при запуске на телефоне",
        "1": "С карты дважды списали оплату за один месяц",
        "2": "Не могу войти в аккаунт, забыл пароль",
        "3": "Хотелось бы иметь тёмную тему оформления",
        "4": "Оператор был груб и совсем не помог",
        "5": "Подозреваю, что мой аккаунт взломали злоумышленники",
        "6": "Как создать первый проект в вашей системе",
        "ambiguous": "Есть вопрос",
    }
    return jsonify(examples)


# ЗАПУСК СЕРВЕРА
if __name__ == "__main__":
    # debug=True — автоматическая перезагрузка при изменении кода
    # port=5000
    # host="0.0.0.0"
    app.run(debug=True, host="0.0.0.0", port=5000)

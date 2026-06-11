import os
import pickle
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dataset import get_texts_and_labels, CATEGORY_LABELS, AMBIGUOUS_EXAMPLES
from model import build_model, predict_with_confidence, VOCAB_SIZE, MAX_LEN

# Параметры
EPOCHS              = 80
BATCH_SIZE          = 32
TEST_SIZE           = 0.2
RANDOM_SEED         = 42
UNCERTAINTY_THRESHOLD = 0.55

SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Загрузка датасета
print("=" * 60)
print("ШАГ 1: Загрузка датасета")
print("=" * 60)

texts, labels = get_texts_and_labels()
labels = np.array(labels)

print(f"Загружено {len(texts)} примеров, {len(CATEGORY_LABELS)} категорий")
for cat_id, cat_name in CATEGORY_LABELS.items():
    count = (labels == cat_id).sum()
    print(f"  [{cat_id}] {cat_name}: {count}")


# Токенизация
print("\n" + "=" * 60)
print("ШАГ 2: Токенизация текстов")
print("=" * 60)

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>",
    lower=True,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN, padding="pre", truncating="pre")

real_vocab = len(tokenizer.word_index)
print(f"Реальный словарь датасета: {real_vocab} токенов")
print(f"Используем top-{VOCAB_SIZE} (запас ×{VOCAB_SIZE/real_vocab:.2f})")
print(f"Форма матрицы X: {X.shape}  → {X.shape[0]} примеров × {X.shape[1]} токенов")


# Разбивка данных
print("\n" + "=" * 60)
print("ШАГ 3: Разбивка данных (stratify=True)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=labels,
)
print(f"Обучающая выборка: {len(X_train)} примеров")
print(f"Тестовая выборка:  {len(X_test)} примеров")


# Построение модели
print("\n" + "=" * 60)
print("ШАГ 4: Построение модели")
print("=" * 60)

model = build_model(
    vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN,
    num_classes=len(CATEGORY_LABELS),
)

model.build(input_shape=(None, MAX_LEN))

model.summary()
print(f"\nПараметров итого: {model.count_params():,}")


# Коллбэки
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",   # следим за точностью
    patience=15,
    restore_best_weights=True,
    mode="max",
    verbose=1,
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,               # lr делится на 2
    patience=7,               # ждём 7 эпох без улучшения
    min_lr=1e-5,
    mode="max",
    verbose=1,
)


# Обучение
print("\n" + "=" * 60)
print("ШАГ 5: Обучение нейросети")
print(f"  epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
      f"обновлений/эпоху≈{len(X_train)//BATCH_SIZE}")
print("=" * 60)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
)

actual_epochs = len(history.history["loss"])
best_val_acc  = max(history.history["val_accuracy"])
print(f"\nОбучение завершено: {actual_epochs} эпох из {EPOCHS}")
print(f"Лучшая val_accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")


# Оценка на тестовой выборке
print("\n" + "=" * 60)
print("ШАГ 6: Оценка на тестовой выборке")
print("=" * 60)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss:     {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")

y_pred_results = predict_with_confidence(
    model, X_test, uncertainty_threshold=UNCERTAINTY_THRESHOLD)
y_pred = np.array([r["predicted_class"] for r in y_pred_results])

category_names = [CATEGORY_LABELS[i] for i in range(len(CATEGORY_LABELS))]
print("\nОтчёт по классам:")
print(classification_report(y_test, y_pred, target_names=category_names))

# Статистика механизма передачи оператору
needs_human = sum(1 for r in y_pred_results if r["needs_human"])
print(f"Передано оператору (уверенность < {UNCERTAINTY_THRESHOLD}): "
      f"{needs_human}/{len(y_pred_results)} "
      f"({needs_human/len(y_pred_results)*100:.1f}%)")


# Проверка на неопределённых обращениях
print("\n" + "=" * 60)
print("ШАГ 7: Проверка механизма передачи оператору")
print("=" * 60)

amb_seqs = tokenizer.texts_to_sequences(AMBIGUOUS_EXAMPLES)
amb_pad  = pad_sequences(amb_seqs, maxlen=MAX_LEN, padding="pre")
amb_res  = predict_with_confidence(
    model, amb_pad, uncertainty_threshold=UNCERTAINTY_THRESHOLD)

for text, result in zip(AMBIGUOUS_EXAMPLES, amb_res):
    status = ("⚠ ОПЕРАТОР" if result["needs_human"]
              else CATEGORY_LABELS[result["predicted_class"]])
    print(f"  «{text}»")
    print(f"    → {status}  (уверенность: {result['confidence']:.0%})")


# Сохранение артефактов
print("\n" + "=" * 60)
print("ШАГ 8: Сохранение артефактов")
print("=" * 60)

# Модель
model_path = os.path.join(SAVE_DIR, "classifier.keras")
model.save(model_path)
print(f"  Модель:     {model_path}")

# Токенайзер
tok_path = os.path.join(SAVE_DIR, "tokenizer.pkl")
with open(tok_path, "wb") as f:
    pickle.dump(tokenizer, f)
print(f"  Токенайзер: {tok_path}")

# Параметры (нужны app.py и visualize.py)
params = {
    "max_len":               MAX_LEN,
    "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
    "vocab_size":            VOCAB_SIZE,
    "random_seed":           RANDOM_SEED,
    "test_accuracy":         float(test_acc),
    "actual_epochs":         actual_epochs,
}
params_path = os.path.join(SAVE_DIR, "params.pkl")
with open(params_path, "wb") as f:
    pickle.dump(params, f)
print(f"  Параметры:  {params_path}")

# История обучения
history_path = os.path.join(SAVE_DIR, "training_history.pkl")
with open(history_path, "wb") as f:
    pickle.dump(history.history, f)
print(f"  История:    {history_path}")

print("\n" + "=" * 60)
print(f"   Обучение завершено!")
print(f"   Точность на тесте:  {test_acc*100:.1f}%")
print(f"   Эпох обучения:      {actual_epochs}")
print("=" * 60)

import os
import pickle
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from dataset import get_texts_and_labels, CATEGORY_LABELS
from model import predict_with_confidence, VOCAB_SIZE, MAX_LEN

# Константы
PLOTS_DIR   = "plots"
SAVE_DIR    = "saved_model"
RANDOM_SEED = 42
THRESHOLD   = 0.55

os.makedirs(PLOTS_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CATEGORY_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#2E86AB", "#937860",
]
SHORT_NAMES = ["Техника", "Оплата", "Аккаунт", "Функции",
               "Жалобы", "Безопасность", "Консульт."]

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "figure.dpi":         120,
})

# Загрузка артефактов
print("Загрузка модели и данных...")
model     = tf.keras.models.load_model(
    os.path.join(SAVE_DIR, "classifier.keras"))
tokenizer = pickle.load(
    open(os.path.join(SAVE_DIR, "tokenizer.pkl"), "rb"))

texts, labels = get_texts_and_labels()
labels = np.array(labels)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN, padding="pre", truncating="pre")

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2,
    random_state=RANDOM_SEED, stratify=labels,
)

results_all = predict_with_confidence(model, X_test, uncertainty_threshold=THRESHOLD)

y_pred      = np.array([r["predicted_class"] for r in results_all])
confidences = [r["confidence"] for r in results_all]
test_acc    = (y_pred == y_test).mean()
print(f"Точность на тесте: {test_acc*100:.1f}%")

# История обучения
history_path = os.path.join(SAVE_DIR, "training_history.pkl")
if os.path.exists(history_path):
    history_data = pickle.load(open(history_path, "rb"))
    print("История обучения загружена.")
else:
    print("ВНИМАНИЕ: training_history.pkl не найден.")
    print("  График 01_learning_curves.png будет пропущен.")
    print("  Убедитесь что train.py сохраняет историю.")
    history_data = None


# ГРАФИК 1: Кривые обучения

if history_data:
    print("\n[1/5] Кривые обучения...")
    train_loss = history_data["loss"]
    val_loss   = history_data["val_loss"]
    train_acc  = history_data["accuracy"]
    val_acc    = history_data["val_accuracy"]
    epochs_r   = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Кривые обучения нейросети",
                 fontsize=15, fontweight="bold", y=1.02)

    # Loss
    ax = axes[0]
    ax.plot(epochs_r, train_loss, color="#4C72B0", linewidth=2,
            label="Обучение", marker="o", markersize=3)
    ax.plot(epochs_r, val_loss,   color="#DD8452", linewidth=2,
            label="Валидация", marker="s", markersize=3, linestyle="--")
    ax.set_title("Функция потерь (Cross-Entropy Loss)")
    ax.set_xlabel("Эпоха"); ax.set_ylabel("Потери")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs_r, train_acc, color="#55A868", linewidth=2,
            label="Обучение", marker="o", markersize=3)
    ax.plot(epochs_r, val_acc,   color="#C44E52", linewidth=2,
            label="Валидация", marker="s", markersize=3, linestyle="--")
    ax.axhline(y=test_acc, color="grey", linestyle=":", linewidth=1.5,
               label=f"Тест: {test_acc*100:.1f}%")
    ax.set_title("Точность классификации (Accuracy)")
    ax.set_xlabel("Эпоха"); ax.set_ylabel("Точность"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "01_learning_curves.png")
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"  → {p}")
else:
    print("\n[1/5] Пропущен (нет training_history.pkl)")


# ГРАФИК 2: Матрица ошибок

print("\n[2/5] Матрица ошибок...")
cm      = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle("Матрица ошибок (Confusion Matrix)",
             fontsize=14, fontweight="bold")
cmap = LinearSegmentedColormap.from_list("blues", ["#f0f4ff", "#1a4ea0"])
im   = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
             label="Доля правильных ответов")

ax.set_xticks(range(len(CATEGORY_LABELS)))
ax.set_yticks(range(len(CATEGORY_LABELS)))
ax.set_xticklabels(SHORT_NAMES, fontsize=9, rotation=20, ha="right")
ax.set_yticklabels(SHORT_NAMES, fontsize=9)
ax.set_xlabel("Предсказанный класс"); ax.set_ylabel("Истинный класс")

for i in range(len(CATEGORY_LABELS)):
    for j in range(len(CATEGORY_LABELS)):
        color = "white" if cm_norm[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                ha="center", va="center", fontsize=8,
                color=color, fontweight="bold")

plt.tight_layout()
p = os.path.join(PLOTS_DIR, "02_confusion_matrix.png")
plt.savefig(p, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ГРАФИК 3: Распределение датасета

print("\n[3/5] Распределение датасета...")
cat_counts = [(CATEGORY_LABELS[i], (labels == i).sum())
              for i in range(len(CATEGORY_LABELS))]
names_p  = [c[0] for c in cat_counts]
counts_p = [c[1] for c in cat_counts]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Распределение датасета по категориям",
             fontsize=14, fontweight="bold")

# Горизонтальные бары
ax = axes[0]
bars = ax.barh(names_p, counts_p, color=CATEGORY_COLORS,
               edgecolor="white", height=0.6)
ax.set_xlabel("Количество примеров")
ax.set_title("Абсолютное количество")
ax.set_xlim(0, max(counts_p) * 1.2)
for bar, count in zip(bars, counts_p):
    ax.text(bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(count), va="center", fontweight="bold")
ax.grid(axis="x", alpha=0.3)

# Кольцевая диаграмма
ax = axes[1]
wedges, texts, autotexts = ax.pie(
    counts_p, labels=names_p, colors=CATEGORY_COLORS,
    autopct="%1.0f%%", startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
)
for t in autotexts:
    t.set_fontweight("bold"); t.set_fontsize(9)
ax.set_title("Доля в процентах")

plt.tight_layout()
p = os.path.join(PLOTS_DIR, "03_dataset_distribution.png")
plt.savefig(p, bbox_inches="tight"); plt.close()
print(f"  → {p}")


# ГРАФИК 4: Уверенность модели

print("\n[4/5] Уверенность модели...")
correct_mask = (y_pred == y_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Уверенность модели при классификации",
             fontsize=14, fontweight="bold")

# Гистограмма: верные vs ошибочные
print("\n[4/5] Уверенность модели...")
correct_mask = (y_pred == y_test)
conf_correct = [c for c, ok in zip(confidences, correct_mask) if ok]
conf_wrong   = [c for c, ok in zip(confidences, correct_mask) if not ok]
 
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Уверенность модели при классификации",
             fontsize=14, fontweight="bold")
 
bins = np.linspace(0, 1, 21)
ax.hist(conf_correct, bins=bins, alpha=0.7, color="#55A868",
        label=f"Верно ({len(conf_correct)})")
ax.hist(conf_wrong,   bins=bins, alpha=0.7, color="#C44E52",
        label=f"Ошибка ({len(conf_wrong)})")
ax.axvline(x=THRESHOLD, color="black", linestyle="--",
           linewidth=1.5, label=f"Порог ({THRESHOLD})")
ax.set_xlabel("Уверенность"); ax.set_ylabel("Кол-во примеров")
ax.set_title("Распределение уверенности")
ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3)
 
plt.tight_layout()
p = os.path.join(PLOTS_DIR, "04_confidence_analysis.png")
plt.savefig(p, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ГРАФИК 5: Анализ гиперпараметров (9 параметров, ceteris paribus)

print("\n[5/5] Анализ гиперпараметров (долго, ~10–20 мин)...")

BASE = {
    "dataset_size":  None,
    "epochs":        40,
    "batch_size":    32,
    "vocab_size":    2000,
    "max_len":       40,
    "embedding_dim": 64,
    "dropout_rate":  0.5,
    "hidden_units":  128,
    "num_classes":   len(CATEGORY_LABELS),
}

PARAM_GRIDS = {
    "dataset_size":  [50, 100, 150, 200, 280, 380, 470, None],
    "epochs":        [5, 10, 20, 40, 60, 80, 120],
    "batch_size":    [4, 8, 16, 32, 64, 128],
    "vocab_size":    [300, 500, 1000, 2000, 3000, 5000],
    "max_len":       [10, 20, 30, 40, 60, 80],
    "embedding_dim": [8, 16, 32, 64, 128, 256],
    "dropout_rate":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "hidden_units":  [16, 32, 64, 128, 256, 512],
    "num_classes":   [2, 3, 4, 5, 6, 7],
}

TITLES_HP = {
    "dataset_size":  "1. Размер датасета",
    "epochs":        "2. Количество эпох",
    "batch_size":    "3. Размер батча",
    "vocab_size":    "4. Размер словаря",
    "max_len":       "5. Длина последовательности",
    "embedding_dim": "6. Размерность эмбеддинга",
    "dropout_rate":  "7. Коэффициент Dropout",
    "hidden_units":  "8. Нейронов в скрытом слое",
    "num_classes":   "9. Количество категорий",
}

XLABELS_HP = {
    "dataset_size":  "Примеров в датасете",
    "epochs":        "Эпохи",
    "batch_size":    "Batch size (примеров)",
    "vocab_size":    "Слов в словаре",
    "max_len":       "Токенов (слов)",
    "embedding_dim": "Размерность вектора слова",
    "dropout_rate":  "Dropout rate (0 = выкл.)",
    "hidden_units":  "Нейронов в Dense-слое",
    "num_classes":   "Количество классов",
}

COLORS_HP = {
    "dataset_size":  "#1565C0",
    "epochs":        "#4C72B0",
    "batch_size":    "#E65100",
    "vocab_size":    "#2E7D32",
    "max_len":       "#B71C1C",
    "embedding_dim": "#6A1B9A",
    "dropout_rate":  "#AD1457",
    "hidden_units":  "#00695C",
    "num_classes":   "#4E342E",
}

TOTAL_N = len(labels)


def _prepare(vocab_size, max_len, num_classes, dataset_size=None):
    t_all, l_all = get_texts_and_labels()
    if num_classes < len(CATEGORY_LABELS):
        pairs  = [(t, l) for t, l in zip(t_all, l_all) if l < num_classes]
        t_all  = [p[0] for p in pairs]
        l_all  = [p[1] for p in pairs]
    if dataset_size is not None:
        per_cls = max(2, dataset_size // num_classes)
        rng = np.random.RandomState(RANDOM_SEED)
        st, sl = [], []
        for cls in range(num_classes):
            idx = [i for i, l in enumerate(l_all) if l == cls]
            chosen = rng.permutation(idx)[:per_cls]
            st += [t_all[i] for i in chosen]
            sl += [l_all[i] for i in chosen]
        t_all, l_all = st, sl
    la = np.array(l_all)
    tok = Tokenizer(num_words=vocab_size, oov_token="<OOV>", lower=True)
    tok.fit_on_texts(t_all)
    seqs = tok.texts_to_sequences(t_all)
    Xd = pad_sequences(seqs, maxlen=max_len, padding="pre", truncating="pre")
    try:
        return train_test_split(Xd, la, test_size=0.2,
                                random_state=RANDOM_SEED, stratify=la)
    except ValueError:
        return train_test_split(Xd, la, test_size=0.2,
                                random_state=RANDOM_SEED)


def _run(param_name, param_value):
    p = dict(BASE); p[param_name] = param_value
    Xtr, Xv, ytr, yv = _prepare(
        p["vocab_size"], p["max_len"], p["num_classes"], p["dataset_size"])
    if len(Xtr) < 10:
        return 1.0 / p["num_classes"]
    m = keras.Sequential([
        layers.Embedding(p["vocab_size"], p["embedding_dim"],
                         embeddings_regularizer=
                             __import__("tensorflow").keras.regularizers.L2(1e-4)),
        layers.SpatialDropout1D(min(p["dropout_rate"], 0.5)),
        layers.GlobalAveragePooling1D(),
        layers.Dense(p["hidden_units"], activation="relu",
                     kernel_regularizer=
                         __import__("tensorflow").keras.regularizers.L2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(p["dropout_rate"]),
        layers.Dense(p["num_classes"], activation="softmax"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(0.001),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = m.fit(Xtr, ytr, epochs=p["epochs"],
                 batch_size=min(p["batch_size"], len(Xtr)),
                 validation_data=(Xv, yv), verbose=0)
    best = max(hist.history["val_accuracy"])
    del m; keras.backend.clear_session()
    return best


hp_results = {}
for pname, grid in PARAM_GRIDS.items():
    print(f"  → {pname}")
    hp_results[pname] = {
        "values": grid,
        "accs":   [_run(pname, v) for v in grid],
    }

# Рисуем 3×3
fig = plt.figure(figsize=(3 * 7.5, 3 * 4.5 + 1.1))
fig.suptitle(
    "Зависимость точности нейросети от параметров обучения и архитектуры\n"
    "Каждый параметр исследуется отдельно при фиксированных остальных (метод ceteris paribus)",
    fontsize=12.5, fontweight="bold", y=0.995, color="#1a1a2e",
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.42)

for idx, pname in enumerate(PARAM_GRIDS):
    ax    = fig.add_subplot(gs[idx // 3, idx % 3])
    color = COLORS_HP[pname]
    vraw  = hp_results[pname]["values"]
    accs  = hp_results[pname]["accs"]
    vx    = [TOTAL_N if v is None else v for v in vraw]
    base_v = BASE[pname]
    base_x = TOTAL_N if base_v is None else base_v
    base_a = accs[vraw.index(base_v)]
    mx, mn = max(accs), min(accs)
    rng_a  = max(mx - mn, 0.05)

    ax.fill_between(vx, [mn] * len(vx), accs, alpha=0.13, color=color)
    ax.plot(vx, accs, color=color, linewidth=2.2,
            marker="o", markersize=6, markerfacecolor="white",
            markeredgewidth=1.8, markeredgecolor=color)
    ax.axhline(mx, color=color, linestyle="--", linewidth=1.0, alpha=0.45)
    ax.text(vx[0], mx + rng_a * 0.09, f"max = {mx:.2f}",
            fontsize=7.5, color=color, fontweight="bold")
    ax.scatter([base_x], [base_a], s=100, facecolors="white",
               edgecolors=color, linewidths=2.5, zorder=5)
    dx = (max(vx) - min(vx)) * 0.03
    ax.annotate(f"база\n{base_x}", xy=(base_x, base_a),
                xytext=(base_x + dx, base_a + rng_a * 0.18),
                fontsize=7, color=color, style="italic",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7))
    if pname == "num_classes":
        ax.plot(vx, [1.0 / v for v in vx], color="#9e9e9e",
                linewidth=1.0, linestyle=":", label="случайное угадывание")
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")

    ax.set_title(TITLES_HP[pname], fontsize=10.5,
                 fontweight="bold", pad=7, color="#1a1a2e")
    ax.set_xlabel(XLABELS_HP[pname], fontsize=8.5, labelpad=4)
    ax.set_ylabel("val_accuracy", fontsize=8.5)
    margin = rng_a * 0.22
    ax.set_ylim(max(0.0, mn - margin), min(1.02, mx + margin + 0.08))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune="both"))
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.grid(axis="x", alpha=0.12, linewidth=0.5)
    ax.set_xticks(vx)
    if len(vx) > 5 or max(vx) >= 500:
        ax.set_xticklabels([str(v) for v in vx],
                           rotation=35, ha="right", fontsize=7.5)
    else:
        ax.set_xticklabels([str(v) for v in vx], fontsize=8)

p = os.path.join(PLOTS_DIR, "05_hyperparam_analysis.png")
plt.savefig(p, bbox_inches="tight", facecolor="white", dpi=130)
plt.close()
print(f"  → {p}")



print("\n" + "=" * 60)
print("  Все графики построены:")
files = [
    "01_learning_curves.png",
    "02_confusion_matrix.png",
    "03_dataset_distribution.png",
    "04_confidence_analysis.png",
    "05_hyperparam_analysis.png",
]
for name in files:
    path   = os.path.join(PLOTS_DIR, name)
    status = "✓" if os.path.exists(path) else "✗ не создан"
    print(f"  {name}  {status}")
print("=" * 60)
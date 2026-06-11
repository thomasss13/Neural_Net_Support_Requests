import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

VOCAB_SIZE    = 2000
MAX_LEN       = 40
EMBEDDING_DIM = 64
NUM_CLASSES   = 7


def build_model(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
                max_len=MAX_LEN, num_classes=NUM_CLASSES):
    
    model = keras.Sequential([
    
        # Embedding с L2-регуляризацией весов
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_regularizer=regularizers.L2(1e-4),
            name="embedding",
        ),
        
        # Агрессивный SpatialDropout: 40% векторов слов обнуляется
        layers.SpatialDropout1D(0.4, name="spatial_dropout"),
        
        # Усредняем все векторы, вектор фиксированной длины
        layers.GlobalAveragePooling1D(name="pooling"),
        
        # Один скрытый слой с L2-регуляризацией на весах
        layers.Dense(
            128, activation="relu",
            kernel_regularizer=regularizers.L2(1e-4),
            name="hidden",
        ),
        
        # Нормализация выходов нейронов внутри батча.
        layers.BatchNormalization(name="batch_norm"),
        
        # Dropout 50% — сильная регуляризация
        layers.Dropout(0.5, name="dropout"),
        
        # Выходной слой: softmax, вероятности по 7 классам
        layers.Dense(num_classes, activation="softmax", name="output"),
    ])

    # Компилируем с оптимизатором Adam
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_with_confidence(model, sequences, uncertainty_threshold=0.55):

    probabilities = model.predict(sequences, verbose=0)
    results = []
    for probs in probabilities:
        predicted_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        results.append({
            "predicted_class": predicted_class,
            "confidence":      round(confidence, 4),
            "all_probs":       [round(float(p), 4) for p in probs],
            "needs_human":     confidence < uncertainty_threshold,
        })
    return results


if __name__ == "__main__":
    model = build_model()
    model.build(input_shape=(None, MAX_LEN))
    model.summary()
    print(f"\nВсего параметров: {model.count_params():,}")

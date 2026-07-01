# -*- coding: utf-8 -*-
"""
Данные для вкладок «Презентация» и «Глоссарий» веб-интерфейса.

GLOSSARY  — словарь терминов: короткое определение + ссылка на статью
            для теоретического углубления.
SLIDES    — метаданные 27 слайдов курсовой работы (заголовок, файл
            изображения, список терминов, встречающихся на слайде).
"""

# ──────────────────────────────────────────────────────────────────
# ГЛОССАРИЙ ТЕРМИНОВ
# ──────────────────────────────────────────────────────────────────
GLOSSARY = {

    "neural_network": {
        "name": "Нейронная сеть",
        "definition": "Модель обработки данных из связанных «нейронов», "
                       "организованных слоями: входной → скрытые → выходной. "
                       "Каждый слой выделяет всё более сложные признаки данных.",
        "url": "https://en.wikipedia.org/wiki/Neural_network_(machine_learning)",
        "source": "Wikipedia",
    },
    "neuron": {
        "name": "Искусственный нейрон",
        "definition": "Упрощённая математическая модель биологического нейрона: "
                       "принимает входные сигналы, взвешивает их и передаёт результат дальше.",
        "url": "https://en.wikipedia.org/wiki/Artificial_neuron",
        "source": "Wikipedia",
    },
    "affine": {
        "name": "Аффинное преобразование (Wx + b)",
        "definition": "Линейное преобразование входа со смещением — основа того, "
                       "как нейрон комбинирует взвешенные входы перед функцией активации.",
        "url": "https://en.wikipedia.org/wiki/Affine_transformation",
        "source": "Wikipedia",
    },
    "relu": {
        "name": "ReLU (функция активации)",
        "definition": "f(x) = max(0, x). Вносит нелинейность в сеть, оставляя "
                       "положительные значения без изменений и обнуляя отрицательные.",
        "url": "https://en.wikipedia.org/wiki/ReLU",
        "source": "Wikipedia",
    },
    "tokenization": {
        "name": "Токенизация",
        "definition": "Разбиение текста на отдельные единицы (слова, токены) "
                       "перед тем, как превратить их в числа для нейросети.",
        "url": "https://en.wikipedia.org/wiki/Lexical_analysis",
        "source": "Wikipedia",
    },
    "padding": {
        "name": "Паддинг последовательностей",
        "definition": "Выравнивание всех последовательностей токенов до одной длины "
                       "нулями — так, чтобы их можно было объединить в единую матрицу.",
        "url": "https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences",
        "source": "TensorFlow docs",
    },
    "embedding": {
        "name": "Эмбеддинг (векторное представление слов)",
        "definition": "Обучаемое представление слова в виде плотного вектора чисел. "
                       "Слова, близкие по смыслу, оказываются близко в этом пространстве.",
        "url": "https://en.wikipedia.org/wiki/Word_embedding",
        "source": "Wikipedia",
    },
    "softmax": {
        "name": "Softmax",
        "definition": "Функция, превращающая набор чисел (логитов) в вероятности, "
                       "которые в сумме дают 1,0. Используется на выходном слое классификатора.",
        "url": "https://en.wikipedia.org/wiki/Softmax_function",
        "source": "Wikipedia",
    },
    "regularization": {
        "name": "L1 / L2-регуляризация",
        "definition": "Штраф за слишком большие веса модели, добавляемый к функции потерь. "
                       "Снижает риск переобучения.",
        "url": "https://en.wikipedia.org/wiki/Regularization_(mathematics)",
        "source": "Wikipedia",
    },
    "cross_entropy": {
        "name": "Cross-Entropy Loss",
        "definition": "Функция потерь для классификации: измеряет, насколько "
                       "предсказанное распределение вероятностей отличается от истинной метки.",
        "url": "https://en.wikipedia.org/wiki/Cross-entropy",
        "source": "Wikipedia",
    },
    "backprop": {
        "name": "Обратное распространение ошибки (backpropagation)",
        "definition": "Алгоритм вычисления градиентов функции потерь по всем весам сети "
                       "с помощью правила цепочки, слой за слоем в обратном направлении.",
        "url": "https://en.wikipedia.org/wiki/Backpropagation",
        "source": "Wikipedia",
    },
    "gradient_descent": {
        "name": "Градиентный спуск / оптимизатор Adam",
        "definition": "Метод обновления весов в направлении убывания функции потерь. "
                       "Adam — адаптивная версия, использующая моменты градиента для более умных шагов.",
        "url": "https://en.wikipedia.org/wiki/Stochastic_gradient_descent",
        "source": "Wikipedia",
    },
    "dropout": {
        "name": "Dropout",
        "definition": "Регуляризация, случайно отключающая долю нейронов на каждом шаге "
                       "обучения — модель не может полагаться на отдельные нейроны.",
        "url": "https://en.wikipedia.org/wiki/Dropout_(neural_networks)",
        "source": "Wikipedia",
    },
    "spatial_dropout": {
        "name": "SpatialDropout1D",
        "definition": "Вариант Dropout, обнуляющий целиком векторы эмбеддингов, а не "
                       "отдельные числа — заставляет модель искать общие паттерны, а не запоминать слова.",
        "url": "https://keras.io/api/layers/regularization_layers/spatial_dropout1d/",
        "source": "Keras docs",
    },
    "batchnorm": {
        "name": "Batch Normalization",
        "definition": "Нормализует активации внутри каждого батча (вычитает среднее, "
                       "делит на стандартное отклонение), ускоряя и стабилизируя обучение.",
        "url": "https://en.wikipedia.org/wiki/Batch_normalization",
        "source": "Wikipedia",
    },
    "early_stopping": {
        "name": "EarlyStopping",
        "definition": "Остановка обучения, если контролируемая метрика (например, "
                       "val_accuracy) не улучшается заданное число эпох подряд.",
        "url": "https://en.wikipedia.org/wiki/Early_stopping",
        "source": "Wikipedia",
    },
    "reduce_lr": {
        "name": "ReduceLROnPlateau",
        "definition": "Колбэк, уменьшающий learning rate, когда метрика перестаёт "
                       "улучшаться — позволяет модели делать более точные шаги ближе к минимуму.",
        "url": "https://keras.io/api/callbacks/reduce_lr_on_plateau/",
        "source": "Keras docs",
    },
    "overfitting": {
        "name": "Переобучение (overfitting)",
        "definition": "Модель слишком хорошо запоминает обучающие примеры, но теряет "
                       "способность обобщать на новые данные. Видно по разрыву train/val accuracy.",
        "url": "https://en.wikipedia.org/wiki/Overfitting",
        "source": "Wikipedia",
    },
    "pooling": {
        "name": "GlobalAveragePooling",
        "definition": "Усредняет вектора всех позиций последовательности в один вектор "
                       "фиксированной длины — теряет порядок слов, но снижает число параметров.",
        "url": "https://keras.io/api/layers/pooling_layers/global_average_pooling1d/",
        "source": "Keras docs",
    },
    "supervised": {
        "name": "Обучение с учителем",
        "definition": "Метод, при котором модель обучается на парах «вход → правильный "
                       "ответ» (размеченные данные), чтобы затем предсказывать ответ самостоятельно.",
        "url": "https://en.wikipedia.org/wiki/Supervised_learning",
        "source": "Wikipedia",
    },
    "confusion_matrix": {
        "name": "Матрица ошибок (confusion matrix)",
        "definition": "Таблица, сопоставляющая истинные и предсказанные классы — показывает, "
                       "какие категории модель путает между собой.",
        "url": "https://en.wikipedia.org/wiki/Confusion_matrix",
        "source": "Wikipedia",
    },
    "hyperparameter": {
        "name": "Гиперпараметр",
        "definition": "Настройка, задаваемая до начала обучения (размер словаря, batch size, "
                       "число нейронов и т.д.), в отличие от весов, которые модель обучает сама.",
        "url": "https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)",
        "source": "Wikipedia",
    },
    "lstm": {
        "name": "LSTM / Bidirectional LSTM",
        "definition": "Рекуррентная архитектура, читающая последовательность с учётом порядка "
                       "и (в двунаправленном варианте) контекста в обе стороны.",
        "url": "https://en.wikipedia.org/wiki/Long_short-term_memory",
        "source": "Wikipedia",
    },
    "fasttext": {
        "name": "fastText",
        "definition": "Библиотека и набор предобученных векторных представлений слов, "
                       "учитывающих морфологию — особенно полезна для языков с богатым словообразованием.",
        "url": "https://en.wikipedia.org/wiki/FastText",
        "source": "Wikipedia",
    },
    "bert": {
        "name": "BERT",
        "definition": "Предобученная трансформер-модель для понимания языка, "
                       "учитывающая контекст слова с обеих сторон предложения.",
        "url": "https://en.wikipedia.org/wiki/BERT_(language_model)",
        "source": "Wikipedia",
    },
    "kfold": {
        "name": "k-fold кросс-валидация",
        "definition": "Метод оценки модели: данные делятся на k частей, модель обучается "
                       "и тестируется k раз на разных разбиениях — даёт более надёжную оценку качества.",
        "url": "https://en.wikipedia.org/wiki/Cross-validation_(statistics)",
        "source": "Wikipedia",
    },
    "lemmatization": {
        "name": "Лемматизация",
        "definition": "Приведение всех словоформ («аккаунт», «аккаунта», «аккаунте») "
                       "к одной базовой форме (лемме) — сокращает словарь и снижает разреженность данных.",
        "url": "https://en.wikipedia.org/wiki/Lemmatization",
        "source": "Wikipedia",
    },
    "xai": {
        "name": "Интерпретируемость модели (XAI, LIME)",
        "definition": "Методы, объясняющие, почему модель приняла конкретное решение — "
                       "например, какие слова повлияли на выбор категории.",
        "url": "https://en.wikipedia.org/wiki/Explainable_artificial_intelligence",
        "source": "Wikipedia",
    },
}


# ──────────────────────────────────────────────────────────────────
# СЛАЙДЫ ПРЕЗЕНТАЦИИ
# ──────────────────────────────────────────────────────────────────
SLIDES = [
    {"number": 1, "section": "Титул", "image": "1.jpeg",
     "title": "Автоматическая категоризация обращений в техподдержку",
     "subtitle": "Нейросеть на TensorFlow / Keras · 7 категорий · веб-сервис на Flask",
     "terms": []},

    {"number": 2, "section": "Введение", "image": "2.jpeg",
     "title": "Содержание и структура проекта",
     "subtitle": "Стек технологий и структура кода: dataset · model · train · app · index.html · visualize",
     "terms": []},

    {"number": 3, "section": "Введение", "image": "3.jpeg",
     "title": "Постановка задачи",
     "subtitle": "Скорость, рост нагрузки и человеческие ошибки — зачем автоматизировать сортировку обращений",
     "terms": ["neural_network"]},

    {"number": 4, "section": "Раздел I", "image": "4.jpeg",
     "title": "Теоретическая база",
     "subtitle": "Нейронные сети · обработка текста · ключевые концепции",
     "terms": []},

    {"number": 5, "section": "Раздел I", "image": "5.jpeg",
     "title": "Что такое нейронная сеть",
     "subtitle": "Биологический прототип, слои нейронов, веса и функция активации ReLU",
     "terms": ["neuron", "affine", "relu"]},

    {"number": 6, "section": "Раздел I", "image": "6.jpeg",
     "title": "Классификация текста: от слов к числам",
     "subtitle": "Токенизация → числовые ID → паддинг → эмбеддинг → Softmax",
     "terms": ["tokenization", "padding", "embedding", "softmax"]},

    {"number": 7, "section": "Раздел I", "image": "7.jpeg",
     "title": "Эмбеддинги: слова как векторы в пространстве",
     "subtitle": "Обучаемое представление слов и L1/L2-регуляризация весов",
     "terms": ["embedding", "regularization"]},

    {"number": 8, "section": "Раздел I", "image": "8.jpeg",
     "title": "Функция потерь и обратное распространение",
     "subtitle": "Прямой проход → Cross-Entropy → backpropagation → обновление весов (Adam)",
     "terms": ["cross_entropy", "backprop", "gradient_descent"]},

    {"number": 9, "section": "Раздел I", "image": "9.jpeg",
     "title": "Регуляризация: борьба с переобучением",
     "subtitle": "Dropout, SpatialDropout1D, L2, BatchNorm и EarlyStopping в комбинации",
     "terms": ["dropout", "spatial_dropout", "regularization", "batchnorm",
               "early_stopping", "overfitting"]},

    {"number": 10, "section": "Раздел II", "image": "10.jpeg",
     "title": "Реализация проекта",
     "subtitle": "Датасет · архитектура · обучение · веб-интерфейс",
     "terms": []},

    {"number": 11, "section": "Раздел II", "image": "11.jpeg",
     "title": "Датасет: 561 обучающий пример",
     "subtitle": "7 категорий, неравное распределение классов и его риски для Softmax",
     "terms": ["softmax"]},

    {"number": 12, "section": "Раздел II", "image": "12.jpeg",
     "title": "Конкретизация задачи",
     "subtitle": "Малый датасет, русский язык, учебная цель — три ограничения, определившие архитектуру",
     "terms": ["supervised"]},

    {"number": 13, "section": "Раздел II", "image": "13.jpeg",
     "title": "Архитектура нейросети",
     "subtitle": "Embedding → SpatialDropout1D → Pooling → Dense → BatchNorm → Dropout → Softmax",
     "terms": ["pooling", "batchnorm", "dropout", "softmax"]},

    {"number": 14, "section": "Раздел II", "image": "14.jpeg",
     "title": "Гиперпараметры и оптимизация",
     "subtitle": "vocab_size, embedding_dim, max_len, hidden units, batch size — как единая система",
     "terms": ["hyperparameter", "embedding", "regularization"]},

    {"number": 15, "section": "Раздел II", "image": "15.jpeg",
     "title": "Как модель обучается: от случайности к точности",
     "subtitle": "Три фазы обучения и колбэки EarlyStopping / ReduceLROnPlateau",
     "terms": ["early_stopping", "reduce_lr", "gradient_descent"]},

    {"number": 16, "section": "Раздел II", "image": "16.jpeg",
     "title": "Веб-сервис",
     "subtitle": "Flask REST API: POST /classify · GET /api/stats — интерфейс, который вы видите на вкладке «Классификатор»",
     "terms": []},

    {"number": 17, "section": "Раздел III", "image": "17.jpeg",
     "title": "Результаты",
     "subtitle": "Кривые обучения · матрица ошибок · анализ гиперпараметров",
     "terms": []},

    {"number": 18, "section": "Раздел III", "image": "18.jpeg",
     "title": "Кривые обучения нейросети",
     "subtitle": "Точность на тесте 77,9 % — виден разрыв train/val accuracy (переобучение)",
     "terms": ["overfitting", "confusion_matrix"]},

    {"number": 19, "section": "Раздел III", "image": "19.jpeg",
     "title": "Анализ результатов классификации",
     "subtitle": "Сильные и слабые категории, главная ошибка модели, работа порога уверенности 0,55",
     "terms": ["confusion_matrix"]},

    {"number": 20, "section": "Раздел III", "image": "20.jpeg",
     "title": "Параметры данных",
     "subtitle": "Влияние размера датасета, числа эпох и размера батча на точность",
     "terms": ["hyperparameter"]},

    {"number": 21, "section": "Раздел III", "image": "21.jpeg",
     "title": "Параметры архитектуры",
     "subtitle": "Размер словаря, длина последовательности, размерность эмбеддинга",
     "terms": ["hyperparameter", "embedding", "tokenization"]},

    {"number": 22, "section": "Раздел III", "image": "22.jpeg",
     "title": "Регуляризация и ёмкость модели",
     "subtitle": "Dropout rate, число нейронов, количество классов",
     "terms": ["dropout", "hyperparameter"]},

    {"number": 23, "section": "Раздел IV", "image": "23.jpeg",
     "title": "Нюансы и ограничения",
     "subtitle": "Слабости датасета · архитектурные ограничения · стратегия улучшения",
     "terms": []},

    {"number": 24, "section": "Раздел IV", "image": "24.jpeg",
     "title": "Нюансы датасета",
     "subtitle": "Лингвистическое пересечение категорий, малый объём, отсутствие лемматизации",
     "terms": ["lemmatization"]},

    {"number": 25, "section": "Раздел IV", "image": "25.jpeg",
     "title": "Технические ограничения",
     "subtitle": "GlobalAveragePooling теряет порядок слов, нет предобученных эмбеддингов, нет интерпретируемости",
     "terms": ["pooling", "embedding", "lstm", "xai", "kfold"]},

    {"number": 26, "section": "Раздел IV", "image": "26.jpeg",
     "title": "Приоритеты доработки",
     "subtitle": "Лемматизация, расширение датасета, fastText RU, Bidirectional LSTM, k-fold, LIME",
     "terms": ["lemmatization", "kfold", "fasttext", "lstm", "xai"]},

    {"number": 27, "section": "Заключение", "image": "27.jpeg",
     "title": "Итоги работы",
     "subtitle": "Точность 77,9 % · механизм передачи оператору · полный стек от датасета до веб-интерфейса",
     "terms": []},
]

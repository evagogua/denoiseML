
# Оценка эффективности тонкой настройки BERT-подобных моделей в задаче детекции фоновой речи

В реальных сценариях использования голосовых помощников пользовательская речь часто сопровождается фоновыми речевыми звуками. После ASR такая смесь преобразуется в единый текст, в котором целевой запрос оказывается перемешан с нерелевантными словами и фразами. На уровне текста это можно формализовать как задачу бинарной разметки токенов:
- `0` — токен относится к целевой речи пользователя;
- `N` — токен относится к фоновой речи и должен быть удалён.


---
## Используемые данные
### Целевая речь
В качестве источника целевых запросов использовался датасет [`clinc/oos-eval`](https://github.com/clinc/oos-eval/tree/master). Это набор пользовательских интентов, включающий 10 доменов и 15 интентов в каждом. Среди доменов: `banking`, `credit_cards`, `travel`, `work`, `home`, `utility`, `small_talk` и др.

Исходный язык датасета — английский. Для работы на русском языке примеры были переведены с помощью модели:
- [`Helsinki-NLP/opus-mt-en-ru`](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru)

### Фоновая речь
В качестве источника шумовых фраз использовался датасет:
- [`Den4ikAI/russian_dialogues`](https://huggingface.co/datasets/Den4ikAI/russian_dialogues).

Он содержит короткие фрагменты разговорной русской речи, не относящиеся к целевому пользовательскому запросу.

### Составление датасета
Синтетический датасет формировался автоматически:
- в каждое исходное предложение добавлялось от 0 до 4 шумовых фраз;
- длина каждой шумовой вставки составляла от 1 до 9 слов;
- токены исходного запроса получали метку `0`;
- токены добавленного шума получали метку `N`.

### Пример структуры данных
```json
{
  "text": ["я", "сломалась", "могу", "я", "получить", "немного", "денег", "это", "жесть", "чтобы", "оплатить", "мой", "счёт"],
  "denoise_labels": ["N", "N", "0", "0", "0", "0", "0", "N", "N", "0", "0", "0", "0"],
  "classification_labels": "pay_bill"
}
```
### Размер и подвыборки
В экспериментах использовались 4 варианта обучающих выборок:
1. `banking + credit_cards`
2. `travel + work`
2. `banking + credit_cards + travel + work` (`bctw`)
3. `all` — полный набор доступных меток

---

## Архитектуры и методы
### 1. Детекция фоновой речи (Token Classification)
Для каждого токена модель предсказывает, принадлежит ли он классу `0` или `N`.

Использованные модели:
- [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base)
- [`google-bert/bert-base-multilingual-uncased`](https://huggingface.co/google-bert/bert-base-multilingual-uncased)

Обе модели дообучались как `AutoModelForTokenClassification` с двумя метками.

### 2. MLM-filtering 
В качестве эксперимента был использован подход по мотивам [Karimi (2024)](https://ahkarimi.github.io/blog/2024/remove-noise-from-text/):
1. каждый токен поочерёдно маскируется;
2. masked language model оценивает вероятность исходного токена в данной позиции;
3. если вероятность ниже порога `threshold`, токен считается маловероятным и удаляется как шум.

---

## Метрики качества
Для оценки использовались следующие метрики:
- **Token Accuracy** — доля верно размеченных токенов;
- **Macro F1** — усреднение по классам без учёта дисбаланса.

---

## Гиперпараметры
### XLM-RoBERTa (Fine-tuning for Token Classification)
- optimizer: `AdamW`
- learning rate: `1e-5`
- weight decay: `0.01`
- warmup ratio: `0.1`
- epochs: `4`
- eval steps: `200`
- metric for best model: `Accuracy`

### mBERT (Fine-tuning for Token Classification)
- optimizer: `AdamW`
- learning rate: `1e-4`
- weight decay: `0.1`
- warmup ratio: `0.1`
- epochs: `1`
- eval steps: `50`
- metric for best model: `Accuracy`

### MLM-filtering 
- criterion: вероятность исходного токена после маскирования
- подобранный порог: `threshold = 0.005`

---

## Результаты

### Основные результаты (Fine-tuning)

#### Этап 1: Детекция фоновой речи

| Модель | Датасет | Token Accuracy | Macro F1 | Manual Test Macro F1 | 
| :--- | :--- | :--- | :--- | :--- |
| **XLM-R FT** | `banking+credit` | 97.35% | 97.33% | 72.62% | 
| **XLM-R FT** | `travel_work` | 96.85% | 96.83% | 66.90% | 
| **XLM-R FT** | `bctw` | 97.49% | 97.48% | 76.10% | 
| **XLM-R FT** | `all` | 97.52% | 97.52% | 60.01% | 
| **mBERT FT** | `banking+credit` | 95.75% | 95.73% | 59.85% |
| **mBERT FT** | `bctw` | 96.02% | 96.01% | 69.58% | 
| **mBERT FT** | `all` | 95.62% | 95.62% | 57.42% | 

#### Этап 2: Классификация интентов (на `bctw`)

| Модель | Тип данных | Intent Accuracy |
| :--- | :--- | :--- |
| **XLM-R FT (Intent)** | Чистые данные | 92.84% |
| **XLM-R FT (Intent)** | Зашумленные данные | 81.82% |
| **XLM-R FT (Intent)** | Очищенные данные (XLM-R-bctw Denoiser) | 92.18% |

### Результаты MLM-filtering

| Модель | Token Accuracy | Macro F1 |
| :--- | :--- | :--- |
| **XLM-R MLM** | 57.13% | 57.05% |
| **mBERT MLM** | 56.28% | 55.81% |

---

## Инструкция по воспроизведению

### 1. Клонирование репозитория и установка зависимостей

```bash
git clone https://github.com/evagogua/denoiseML
cd denoiseML
pip install -r requirements.txt
```

### 2. Запуск экспериментов

Эксперименты разделены на три Jupyter Notebook.

#### Ноутбук 1: `dataset_builder.ipynb`

Этот ноутбук отвечает за подготовку данных.

**Действия:**
1.  Клонирует репозиторий и устанавливает зависимости.
2.  Загружает и переводит датасет `clinc/oos-eval` с английского на русский.
3.  Загружает датасет `Den4ikAI/russian_dialogues` для создания "шума".
4.  Генерирует зашумленные версии предложений, вставляя случайные фразы из диалогов.
5.  Создает файлы с данными для обучения (`noise_data_*.json`).

#### Ноутбуки 2 и 3: `02_training_XLM_R.ipynb` и `03_traing_mBERT.ipynb`

Эти ноутбуки обучают модели для **детекции фоновой речи (Token Classification)**.

**Действия:**
1.  Загружает предобученную модель (`xlm-roberta-base` или `bert-base-multilingual-uncased`).
2.  Загружает подготовленные данные из `data/noise_data_*.json`.
3.  Производит тонкую настройку (`fine-tuning`) модели для токенной классификации.
4.  Сохраняет обученные модели.

#### Ноутбук 4: `04_mlm_filtering.ipynb`
Этот ноутбук реализует альтернативный подход к очистке текста — MLM-фильтрацию (по мотивам Karimi, 2024).

**Действия**:

1. Загружается предобученная MLM.
2. На валидационной выборке подбирается оптимальный порог `threshold`, максимизирующий метрики качества очистки.
3. Загружается зашумленный датасет (`noise_data_bctw.json`) и применяется MLM-фильтрация с подобранным порогом.
4. Оцениваются результаты очистки MLM-фильтрацией.

#### Ноутбук 5: `05_classification.ipynb`

Этот ноутбук обучает модель **классификации интентов (Sequence Classification)** для оценки качества классификации зашумленных и очищенных от шума интентов.

**Действия:**
1.  **Обучение на чистых данных:**
    *   Загружает `XLM-RoBERTa`.
    *   Загружает чистые данные (`clean_data_bctw.json`) и дообучает модель для классификации интентов.
    *   Сохраняет модель как `class_model`.
2.  **Оценка на зашумленных данных:**
    *   Загружает обученный классификатор интентов.
    *   Загружает зашумленные данные (`noise_data_bctw.json`) и оценивает точность классификации на нем (как бейзлайн).
3.  **Оценка на очищенных данных:**
    *   Загружает лучшую модель денойзера из предыдущего шага.
    *   Использует эту модель для очистки зашумленного датасета (`noise_data_bctw.json`), удаляя "шумовые" токены.
    *   Подает очищенные тексты на вход обученному классификатору интентов и оценивает итоговую точность.

### 3. Использование готовых моделей

Все обученные модели доступны в коллекции на Hugging Face Hub: [**Коллекция denoiseML**](https://huggingface.co/collections/evagogua/denoiseml)

Коллекция включает следующие модели:

**Модели для детекции фоновой речи (XLM-RoBERTa):**
- [`evagogua/banking_credit_denoise_model`](https://huggingface.co/evagogua/banking_credit_denoise_model) — обучена на датасете `banking + credit_cards`.
- [`evagogua/travel_work_denoise_model`](https://huggingface.co/evagogua/travel_work_denoise_model) — обучена на датасете `travel + work`
- ⭐️ [`evagogua/bctw_denoise_model`](https://huggingface.co/evagogua/bctw_denoise_model) ⭐️ — обучена на комбинированном датасете `banking + credit_cards + travel + work`.
- [`evagogua/all_denoise_model`](https://huggingface.co/evagogua/all_denoise_model) — обучена на полном наборе данных.

**Модели для детекции фоновой речи (mBERT):**
- [`evagogua/banking_credit_denoise_google_model`](https://huggingface.co/evagogua/banking_credit_denoise_google_model)
- [`evagogua/bctw_denoise_google_model`](https://huggingface.co/evagogua/bctw_denoise_google_model)
- [`evagogua/full_denoise_google_model`](https://huggingface.co/evagogua/full_denoise_google_model)

**Модель для классификации интентов:**
- [`evagogua/class_model`](https://huggingface.co/evagogua/class_model) — классификатор на 151 интент.


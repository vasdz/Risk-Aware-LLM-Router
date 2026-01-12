# 🛡️ Risk-Aware LLM Router (White-Box Probing)

[English](#english) | [Русский](#russian)

---

## <a name="english"></a>English

### 🚀 Overview
This project implements a unique **White-Box Risk-Aware Routing System** for Large Language Models (LLMs). Unlike standard "LLM-as-a-judge" routers, this system analyzes the internal **hidden states** of a model to predict the probability of hallucination or error *before* generation is complete.

It utilizes **Conformal Risk Control (CRC)** to provide mathematical guarantees on the error rate (e.g., "95% confidence that the model will not hallucinate").

### ✨ Key Features (Why it's unique)
*   **White-Box Probing:** Extracts features directly from the neural network's layers (hidden states), making it significantly faster and cheaper than prompting a second LLM.
*   **Conformal Prediction:** Implements rigorous statistical methods (CRC, CP) to strictly control the risk level ($\alpha$).
*   **Resource Efficient:** Designed to run on consumer hardware (e.g., NVIDIA GTX 1650 Ti) by optimizing inference and using lightweight probes (Logistic Regression / MLP).
*   **Linux/WSL Native:** Fully optimized for Linux environments.

### 🛠️ Tech Stack
*   **Core:** Python 3.10+, PyTorch, Transformers (Hugging Face)
*   **Probing:** Scikit-learn (Logistic Regression, Isotonic Regression for calibration)
*   **API:** FastAPI (for the router service)
*   **Model:** Qwen 2.5 (3B quantized) as the base model

### ⚡ Quick Start (Linux)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the router service
python router_service.py
<a name="russian"></a>Russian
🚀 Описание проекта
Это реализация уникальной системы умной маршрутизации запросов (White-Box Router) для больших языковых моделей. В отличие от стандартных решений, где одна нейросеть спрашивает другую, этот проект анализирует внутренние скрытые состояния (hidden states) модели.

Система "просвечивает" нейросеть рентгеном и с помощью Conformal Risk Control (CRC) дает математическую гарантию надежности ответа. Если риск ошибки высок, запрос перенаправляется или отклоняется.

✨ Почему это круто?
White-Box подход: Мы смотрим внутрь "мозга" модели, а не просто анализируем текст. Это дает высокую точность детекции галлюцинаций.

Математическая гарантия: Используется метод конформного предикшена. Мы не просто "гадаем", а гарантируем уровень риска (например, не более 5% ошибок).

Эффективность: Система оптимизирована для работы на доступном железе (тестировалось на GTX 1650 Ti). "Проб" (детектор) работает мгновенно и не нагружает GPU.

Linux First: Разработано и протестировано в среде Linux / WSL2.

📂 Структура
router_service.py — API сервис роутинга.

03_train_probe.py — обучение детектора ошибок (Probing).

04_conformal_risk.py — расчет математических порогов риска.

gate.py — логика принятия решений (Gate).

🔧 Запуск
Проект полностью готов к работе в среде Linux.

bash
python router_service.py
text

***

### 2. Как загрузить код (Инструкция)

Внимательно выполни эти команды в терминале (в папке с проектом), чтобы **исключить** тяжелые папки `models` и `runs`.

#### Шаг 1: Инициализация и настройка игнора
Если ты уже пробовал `git init`, на всякий случай сбросим всё, чтобы начать чисто.

1.  **Сброс (если нужно)**:
    ```bash
    rm -rf .git  # Удаляем старую историю, если она была "грязной"
    git init     # Создаем новый пустой гит
    ```

2.  **Создание правильного `.gitignore` (Самое важное!)**:
    Создай файл `.gitignore` (или открой существующий) и убедись, что в нем есть эти строки:
    ```text
    .idea/
    __pycache__/
    *.pyc
    .env
    venv/
    
    # Игнорируем тяжелые веса и логи запусков
    models/
    runs/
    
    # Если есть папки с данными
    *.jsonl
    !train.jsonl  # (опционально: можно оставить маленькие файлы, если нужно)
    ```

#### Шаг 2: Добавление файлов и отправка
Теперь, когда `models` и `runs` в игноре, можно смело добавлять всё остальное.

1.  **Добавляем файлы**:
    ```bash
    git add .
    ```

2.  **Проверка (Обязательно)**:
    Напиши `git status`. Ты **НЕ** должен видеть там тысячи файлов из папки `runs` или `models`. Только `.py` файлы, `README.md`, `requirements.txt` и т.д.

3.  **Коммит и пуш**:
    ```bash
    git commit -m "Initial commit: White-box LLM Router core logic"
    git branch -M main
    git remote add origin https://github.com/vasdz/Risk-Aware-LLM-Router.git
    git push -u origin main
    ```
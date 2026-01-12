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
<a name="russian"></a>Русский
🚀 Описание проекта
Это реализация уникальной системы умной маршрутизации запросов (White-Box Router) для больших языковых моделей. В отличие от стандартных решений, где одна нейросеть спрашивает другую, этот проект анализирует внутренние скрытые состояния (hidden states) модели.

Система "просвечивает" нейросеть рентгеном и с помощью Conformal Risk Control (CRC) дает математическую гарантию надежности ответа. Если риск ошибки высок, запрос перенаправляется или отклоняется.

✨ Почему это круто?
White-Box подход: Мы смотрим внутрь "мозга" модели, а не просто анализируем текст. Это дает высокую точность детекции галлюцинаций.

Математическая гарантия: Используется метод конформного предикшена. Мы не просто "гадаем", а гарантируем уровень риска (например, не более 5% ошибок).

Эффективность: Система оптимизирована для работы на доступном железе (тестировалось на NVIDIA GTX 1650 Ti). "Проб" (детектор) работает мгновенно и не нагружает GPU.

Linux First: Разработано и протестировано в среде Linux / WSL2.

🛠️ Стек технологий
Ядро: Python 3.10+, PyTorch, Transformers (Hugging Face)

Анализ (Probing): Scikit-learn (Логистическая регрессия, Изотоническая регрессия для калибровки)

API: FastAPI (сервис маршрутизации)

Базовая модель: Qwen 2.5 (3B quantized)

📂 Структура проекта
router_service.py — API сервис роутинга.

03_train_probe.py — обучение детектора ошибок (Probing).

04_conformal_risk.py — расчет математических порогов риска.

gate.py — логика принятия решений (Gate).

⚡ Быстрый старт (Linux)
bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск сервиса роутинга
python router_service.py
text

### Финальный штрих (обновление на сервере)
После того как замените текст, выполните:
```bash
git add README.md
git commit -m "Final fix: cleanup README formatting"
git push
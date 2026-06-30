# 🏠 House Price Prediction

---

## English

### Overview
A simple machine learning project that predicts house prices using the Random Forest algorithm. The model is trained on house data and demonstrates basic data preprocessing and model evaluation techniques.

### Dataset
The project uses `housedata.csv` containing information about properties including:
- Area (in square feet)
- Number of rooms
- Build year
- Property type (Apartment, House, Duplex, Villa)
- Location
- Street type
- Furnishing status
- Pool availability
- Price (target variable)

### Model
- **Algorithm**: Random Forest Regressor
- **Parameters**: 100 estimators, max depth of 20
- **Evaluation Metrics**: Mean Absolute Error (MAE), R² Score, Cross-Validation Score

### Key Features
- Data cleaning and handling missing values (median/mean imputation)
- Feature encoding (categorical → numerical)
- One-hot encoding for categorical features
- Train-test split (80/20)
- 5-fold cross-validation
- Price prediction function for new properties

### Requirements
```
pandas
scikit-learn
scipy
```

### Installation
```bash
pip install pandas scikit-learn scipy
```

### Usage
```bash
python house.py
```

The script will:
1. Load and preprocess the house data
2. Train the Random Forest model
3. Display cross-validation scores
4. Show MAE and R² metrics
5. Make a test prediction

### Output
The script prints:
- First 5 rows of the dataset
- Dataset shape and statistics
- Model information
- Cross-validation scores
- Mean Absolute Error
- R² Score
- Test predictions

---

## Русский

### Обзор
Простой проект машинного обучения, который прогнозирует цены на дома с помощью алгоритма Random Forest. Модель обучена на данных домов и демонстрирует основные методы предварительной обработки данных и оценки моделей.

### Набор данных
Проект использует `housedata.csv` содержащий информацию о свойствах, включая:
- Площадь (в квадратных футах)
- Количество комнат
- Год постройки
- Тип недвижимости (Квартира, Дом, Дуплекс, Вилла)
- Местоположение
- Тип улицы
- Статус отделки
- Наличие бассейна
- Цена (целевая переменная)

### Модель
- **Алгоритм**: Random Forest Regressor
- **Параметры**: 100 деревьев, максимальная глубина 20
- **Метрики оценки**: Mean Absolute Error (MAE), R² Score, Cross-Validation Score

### Основные возможности
- Очистка данных и обработка пропущенных значений (заполнение медианой/средним)
- Кодирование признаков (категориальные → числовые)
- One-hot кодирование для категориальных признаков
- Разделение на обучающую и тестовую выборки (80/20)
- 5-кратная перекрёстная валидация
- Функция прогнозирования цены для новых объектов

### Требования
```
pandas
scikit-learn
scipy
```

### Установка
```bash
pip install pandas scikit-learn scipy
```

### Использование
```bash
python house.py
```

Скрипт выполнит:
1. Загрузку и предварительную обработку данных о домах
2. Обучение модели Random Forest
3. Вывод оценок перекрёстной валидации
4. Отображение метрик MAE и R²
5. Прогноз для тестового примера

### Результаты
Скрипт выводит:
- Первые 5 строк набора данных
- Форму и статистику данных
- Информацию о модели
- Оценки перекрёстной валидации
- Среднюю абсолютную ошибку
- R² Score
- Тестовые прогнозы

---

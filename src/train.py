import torch # Основная библиотека PyTorch для работы с нейронными сетями
import pandas as pd # Для работы с табличными данными
from model import ChocolateNN # Модель нейронной сети
from sklearn.preprocessing import StandardScaler # Для нормализации данных
import joblib # Для сохранения нормализатора
from torch.utils.data import DataLoader, TensorDataset # Для удобной работы с данными в PyTorch
import matplotlib.pyplot as plt  # Добавляем импорт для визуализации
import numpy as np

def train_model():
    # 1. Загрузка данных
    try:
        data = pd.read_csv("data/chocolate_recipes.csv") # Загружаются данные из файла с рецептами шоколада
    except FileNotFoundError:
        print("Ошибка: Файл data/chocolate_recipes.csv не найден")
        return

    print("Данные успешно загружены. Пример данных:")
    print(data.head())

    # 2. Подготовка данных
    X = data[['cocoa_butter', 'milk_fat', 'sugar', 'lecithin', 'cocoa_solids', 'use_type_code']] # Признаки (ингредиенты и тип использования)
    y = data[['melt_temp', 'cool_temp', 'stab_temp', 'time_melt', 'time_cool', 'time_stab']] # Целевые переменные (температуры и времена процессов)

    # 3. Нормализация данных
    scaler = StandardScaler() # Признаки нормализуются и нормализатор сохраняется для использования при предсказании
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')
    print("Нормализатор сохранен в scaler.pkl")

    # 4. Разделение на train/val и преобразование в тензоры PyTorch
    from sklearn.model_selection import train_test_split

    # Разделение данных (80% train, 20% val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y.values,
        test_size=0.2,
        random_state=42
    )

    # Создание TensorDataset и DataLoader для тренировочных данных
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Создание TensorDataset и DataLoader для валидационных данных
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val))
    loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # shuffle=False для валидации!

    print(f"Данные разделены: {len(train_dataset)} тренировочных, {len(val_dataset)} валидационных")


    # 5. Инициализация модели
    model = ChocolateNN(X.shape[1])

    # 6. Настройка обучения
    criterion = torch.nn.MSELoss() # Функция потерь MSE (среднеквадратичная ошибка)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Оптимизатор Adam с learning rate 0.01

    # 7. Процесс обучения с валидацией
    print("Начало обучения...")

    # Создаем списки для хранения истории ошибок
    train_loss_history = [] #Список для сохранения значений ошибки на обучающей выборке после каждой эпохи
    val_loss_history = [] #Список для сохранения значений ошибки на валидационной выборке после каждой эпохи

    for epoch in range(100): # 100 эпох, после каждой эпохи выводится средняя ошибка (loss) на всех данныъ
        model.train() # Переводим модель в режим обучения
        train_loss = 0 #Накопительная переменная для подсчета ошибки на обучающей выборке за эпоху
        for inputs, targets in loader: # inputs - Батч входных данных (ингредиенты шоколада)
            # targets -  Батч истинных значений (температуры и времена процессов)
            optimizer.zero_grad() # Обнуляем градиенты перед новым батчем
            outputs = model(inputs) # Прямой проход
            loss = criterion(outputs, targets) # Вычисление ошибки между предсказаниями текущего батча (outputs) и истинными значениями (targets)
            loss.backward() # Обратное распространение ошибки
            optimizer.step() # Обновление весов модели
            train_loss += loss.item()

        # Валидация
        model.eval() #Переводим модель в режим оценки
        val_loss = 0 #Накопительная переменная для подсчета ошибки на валидационной выборке за эпоху
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        # Сохраняем ошибки для графиков
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(loader)
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        print(f'Epoch {epoch + 1}/100 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

    # 8. Сохранение модели
    torch.save(model.state_dict(), 'chocolate_model.pth') # Веса обученной модели сохраняются в файл
    print("Модель успешно сохранена в chocolate_model.pth")

    # 9. Визуализация результатов обучения
    plt.figure(figsize=(15, 5))

    # График: Кривая обучения
    plt.plot(train_loss_history, label='Ошибка на тренировочной выборке')
    plt.plot(val_loss_history, label='Ошибка на валидационной выборке')
    plt.title('Кривая обучения: Ошибка на тренировочном и валидационном наборе')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE ошибка')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_analysis.png')  # Сохраняем графики в файл
    plt.show()
    print("Анализ обучения сохранен в training_analysis.png")

    # Оценка модели: сравнение предсказаний и реальных значений
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            all_preds.append(outputs.numpy())
            all_true.append(targets.numpy())

    preds = np.vstack(all_preds)
    true = np.vstack(all_true)

    #Визуализация результатов:  График предсказание vs истинное значение
    target_names = ['melt_temp', 'cool_temp', 'stab_temp', 'time_melt', 'time_cool', 'time_stab']
    plt.figure(figsize=(18, 10))
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        ax.scatter(true[:, i], preds[:, i], alpha=0.5, label='Предсказания')
        ax.plot([true[:, i].min(), true[:, i].max()],
                 [true[:, i].min(), true[:, i].max()],
                 'r--', label='Идеальное совпадение')
        ax.set_xlabel('Истинные значения')
        ax.set_ylabel('Предсказанные значения')
        ax.set_title(f'{target_names[i]}')
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True)
        ax.set_aspect('auto')  # Обеспечивает адекватное масштабирование


    plt.tight_layout(pad=3.0)
    plt.savefig('evaluation_results.png')
    plt.show()
    print("Оценка модели сохранена в evaluation_results.png")

    # 12. Расчёт числовых метрик
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    print("\nМетрики качества по каждому выходному параметру:")
    for i, name in enumerate(target_names):
        mse = mean_squared_error(true[:, i], preds[:, i])
        mae = mean_absolute_error(true[:, i], preds[:, i])
        r2 = r2_score(true[:, i], preds[:, i])
        print(f"{name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²:  {r2:.4f}")


if __name__ == "__main__": # При запуске скрипта напрямую вызывается функция train_model()
    train_model()
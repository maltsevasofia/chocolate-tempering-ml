import logging # Для логирования работы бота
from telegram import Update # Библиотека для работы с Telegram API.
from telegram.ext import Application, CommandHandler, ContextTypes
import torch # Фреймворк для работы с нейронными сетями (PyTorch)
from model import ChocolateNN # Модель нейросети (импортируется из model.py)
import joblib # Для загрузки нормализатора данных (scaler.pkl)
import numpy as np # Для работы с числовыми данными

# Настройка логгирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO) # Настраивает формат логов (время, имя логгера, уровень важности, сообщение)
# Уровень INFO выводит все сообщения, кроме отладочных
logger = logging.getLogger(__name__) # Автоматически подставляет имя текущего модуля

# Загрузка модели и нормализатора
device = torch.device("cpu")
model = ChocolateNN(6).to(device) # Cоздаем экземпляр модели с 6 входными параметрами
model.load_state_dict(torch.load('chocolate_model.pth')) # Загружаем веса обученной модели из файла
model.eval() # Переводит модель в режим предсказания
scaler = joblib.load('scaler.pkl') # Загружает нормализатор для входных данных.


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
    🍫 Бот для темперирования шоколада 🍫

    Введите состав шоколада в формате:
    /temper какао_масло(%) молочный_жир(%) сахар (%) лецитин (1 – есть, 0 – нет) какао_твердое (%) применение (molding - создание шоколадных изделий, coating - создание глазури)

    Пример:
    /temper 40 5 45 1 20 molding
    """
    await update.message.reply_text(help_text)


async def temper(update: Update, context: ContextTypes.DEFAULT_TYPE): # Команда \temper
    try:
        # Парсинг аргументов
        args = context.args
        if len(args) != 6:
            raise ValueError("Нужно 6 параметров!")

        cocoa_butter = float(args[0]) # % какао-масла
        milk_fat = float(args[1]) # % молочного жира
        sugar = float(args[2]) # % сахара
        lecithin = int(args[3]) # 0 или 1 (есть ли лецитин)
        cocoa_solids = float(args[4]) # % какао-твердого
        use_type = args[5].lower() # molding" или "coating"

        # Проверка входных данных
        if use_type not in ['molding', 'coating']:
            raise ValueError("Тип применения должен быть 'molding' или 'coating'")

        # Подготовка данных для модели
        use_code = 1 if use_type == 'molding' else 0
        input_data = np.array([[cocoa_butter, milk_fat, sugar, lecithin, cocoa_solids, use_code]])
        input_scaled = scaler.transform(input_data) # Нормализация
        input_tensor = torch.FloatTensor(input_scaled).to(device)

        # Получение предсказания
        with torch.no_grad(): # Отключает расчет градиентов (ускоряет предсказание)
            output = model(input_tensor).cpu().numpy()[0] # Результат преобразуется в массив NumPy

        # Форматирование результата
        result = {
            'melting_temp': round(output[0], 1), # Температура плавления
            'cooling_temp': round(output[1], 1), # Температура охлаждения
            'stabilization_temp': round(output[2], 1), # Стабилизация
            'melting_time': int(output[3]), # Время плавления (мин)
            'cooling_time': int(output[4]), # Время охлаждения (мин)
            'stabilization_time': int(output[5]) # Время стабилизации (мин)
        }

        # Формирование ответа
        response = f"""
        🔍 Результат для состава:
        Какао-масло: {cocoa_butter}%
        Молочный жир: {milk_fat}%
        Сахар: {sugar}%
        Лецитин: {'да' if lecithin else 'нет'}
        Какао-твердое: {cocoa_solids}%
        Применение: {use_type}

        🌡 Параметры темперирования:
        1. Нагрев до: {result['melting_temp']:.1f}°C ({result['melting_time']} мин)
        2. Охлаждение до: {result['cooling_temp']:.1f}°C ({result['cooling_time']} мин)
        3. Стабилизация при: {result['stabilization_temp']:.1f}°C ({result['stabilization_time']} мин)
        """

        await update.message.reply_text(response)

    except Exception as e: #Обработка ошибок (ловит все исключения)
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)}\nИспользуйте /help для справки")


def main():
    # Создаем приложение (экземпляр бота) и передаем токен бота
    application = Application.builder().token("7164888625:AAHUFGJPX_s-sMwPp8A5GJ-sK1i1MlrW408").build()

    # Регистрируем обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("temper", temper))

    # Запускаем бота в режиме опроса сервера Telegram
    application.run_polling()


if __name__ == '__main__':
    main()
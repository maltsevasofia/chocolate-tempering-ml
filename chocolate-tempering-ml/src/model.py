import torch # Основная библиотека PyTorch
import torch.nn as nn # Модуль для работы с нейронными сетями


class ChocolateNN(nn.Module): # Создаём класс нейронной сети, наследуясь от базового класса nn.Module
    def __init__(self, input_size): # Инициализация
        super(ChocolateNN, self).__init__()
        #Создаем 4 слоя
        self.fc1 = nn.Linear(input_size, 128) # Входной слой 128 нейронов
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 6) # 6 выходных параметров


    def forward(self, x): #Прямой проход
        x = torch.relu(self.fc1(x)) # Входные данные х проходят через fc1 + активация ReLU
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x) # Обычно не используют активацию на выходном слое
        return x
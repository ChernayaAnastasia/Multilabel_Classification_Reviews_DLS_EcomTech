# Задача 

Для каждого отзыва (пользовательские ответы на вопрос о сервисе Самокат) предсказать все классы затрагиваемых тематик. Отзыв может относиться сразу к нескольким классам. Всего тематик 50. Задача решалась в рамках [интенсива "NLP интенсив DLS and Ecom.tech"](https://ods.ai/competitions/dls_ecomtech)

# Данные 

4623 строк (каждая строка - отдельный отзыв) 

5173 строк после аугментации данных (для обучения финальной модели использовалось 85%, для валидации - 15%)


Дисбаланс классов:

![](https://github.com/ChernayaAnastasia/Screenshots/blob/master/Screenshot%202024-10-11%20004139.png)

# Метрика
Accuracy (считается через полное совпадение списка выбранных классов для каждого экземпляра):

from sklearn.metrics import accuracy_score

accuracy_score(all_labels, all_preds)

# Архитектура модели: 

BERT Transformer Classifier (c предобученными эмбеддингами ai-forever/ruBert-base)

Для получения эмбеддингов отзывов использую предобученную модель BERT ai-forever/ruBert-base. Для улучшения представления входных данных делаю конкатенации скрытых состояний последних 4 слоев. После конкатенации скрытых состояний для выполнения классификации использую несколько дополнительных слоев:
* Полносвязный слой (pre_classifier): принимает на вход результат конкатенации скрытых состояний размером config.hidden_size * 4 и проецирует его на пространство меньшей размерности (768 признаков).
* Dropout: чтобы предотвратить переобучение и добавить регуляризацию (в моем случае был dropout rate 0.1)  
* Финальный классификационный слой: линейная проекция, которая отображает 768 признаков на количество классов, заданное в конфигурации модели.

  *Настройки*:
  
    "lr": 2e-4,
    "n_epochs": 20,
    "weight_decay": 1e-06,
    "batch_size": 64

  

# Аугментация тренировочных данных
Выполнена генерация отзывов, принадлежащих к классам, доля которых менее 1% в исходном датасете, с помощью автоматизированной генерации промтов для GPT-4o mini в диалоговом окне браузера. 

# Предобработка
Предобработаны эмодзи, убраны лишние пробелы и вставлены недостающие, применен Яндекс.Спеллер API для автоматической проверки и исправления орфографических ошибок, а также все тексты отзывов приведены к нижнему регистру.

## Ссылки

[Файл с EDA и аугментацией в гугл колаб](https://colab.research.google.com/drive/1cEi2UBUFblA0AvLXDqcTr5SZm6Mw1mEK?usp=sharing)

[Файл с обучением модели и предсказанием в гугл колаб](https://colab.research.google.com/drive/1SUErr6RuWoyCGlJqPRqTYVgE9NpOll9Q?usp=sharing)

[Открыть в nbviewer тетрадку с EDA и аугментацией данных](https://nbviewer.org/github/ChernayaAnastasia/Multilabel_Classification_Reviews_DLS_EcomTech/blob/main/augment_data_ecom_dls.ipynb)

[Открыть в nbviewer тетрадку с обучением модели и предсказанием](https://nbviewer.org/github/ChernayaAnastasia/Multilabel_Classification_Reviews_DLS_EcomTech/blob/main/multilabel_classifier_ecom_dls.ipynb)


## Финальный результат на leaderboard 
Accuracy score: 0.5274261603 (private)

Accuracy score: 0.5223847518 (public)


## Автор
**Chernaya Anastasia** - [Telegram](https://t.me/ChernayaAnastasia)

## Сертификат
[Сертификат призера](https://github.com/ChernayaAnastasia/Certificates/blob/main/%D0%90%D0%BD%D0%B0%D1%81%D1%82%D0%B0%D1%81%D0%B8%D1%8F%20%D0%A7%D0%B5%D1%80%D0%BD%D0%B0%D1%8F.pdf)


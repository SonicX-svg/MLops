# MLops_labs
### Учебные проекты MLops
## MLops-1
<details>
<summary>Click to expand</summary>  <br>
### Скрипты по созданию, обработке, обучению и предсказанию<br>
1. data_creation.py - генерирует фаилы train и test с данными для предскаязания хорошего дня:<br>
      - температура<br>
      - пол<br>
      - возраст<br>
      - осадки<br>
      - хороший день  <br>
   В данные включены пропуски выбросы и шум. Корреляции между признаками нет, поэтому никаких выводов сделать нельзя  <br>
2. model_preprocessing.py - получает сгенерированные данные и выполняет предобработку:<br>
   1. удаляем пропущенные значени - строки с 1 непустым<br>
   2. разделяем датасет на качественные и количественные<br>
   3. работаем с количственными: 3.1 выявляем качественные в количественных 3.2 работаем с пропусками 3.3 работа с аномальными значениями 3.4. масштабирование<br>
   4. работаем с качественными данными: 4.1 точно также работам с пропусками и выбросами 4.2 sklearn.preprocessing.LabelEncoder<br>
   5. совмещаем обработанные кат и чис данные обратно<br>
3. model_preparation.py - фаил создает модель логистической регрессии, которая обучается на train.csv и для последующего применения модели, сохраняется с помощью pickle (model.pkl)<br>
4. model_testing.py - тестирует модель и выводит метрики качества.<br>
</details>

## MLops-2
<details>
<summary>Click to expand</summary>
### Скрипты по созданию, обработке, обучению и предсказанию с применением  Jenkins<br>
Мдель классификации данных о гистологических параметрах опухолей молочной железы на основе метода knn.<br>
1. Проведен разведочный анаализ и соотвесвующая предобработка:<br>
   - пропущенные значения<br>
   - количественные и качественные переменные<br>
   - нормализация<br>
   - отбор наиболее значимых параметров по наибольшей разнице в средних значениях. Для улчшения модели можно было использовать методы уменьшения размерности.<br>
2. Создана и обучена модель на методе knn.<br>
3. Приведены метри качества работы модели. Точность модели достаточно высокая, но accuracy не дает информации о ложноотрицательных случаях, которые наиболее важные при постановке диагнозов. <br>
</details>

## MLops-3 (lab3)
<details>
<summary>Click to expand</summary>
### Скрипты по созданию, обработке, обучению и предсказанию с применением  контейнеризации Docker<br>
Мдель классификации данных Iris LogisticRegression<br>
1. Проведен разведочный анаализ и соотвесвующая предобработка:<br>
   - пропущенные значения<br>
   - количественные и качественные переменные<br>
   - нормализация<br>
   - отбор наиболее значимых параметров по наибольшей разнице в средних значениях. Для улчшения модели можно было использовать методы уменьшения размерности.<br>
2. Создана и обучена модель на методе LogisticRegression<br>
3. Приведены метри качества работы модели. <br>
</details>

## MLops-4 (lab4)
<details>
<summary>Click to expand</summary>
### logfile и Gdrive_disk с демонстрацией работы с утилитой контроля версий dvc на наборе данных titanic <br>
Фаилы изменения датасета: change_data.py & change_data2.py
1. Инициализируется git и dvc. 
2. Дбавлен фаил с данными для отслеживания. (dvc add lab4)<br>
3. Создан и добавлен гугл диск. <br>
4. Проведены изменения и их пуши. <br>
5. Проведены переходы по версиям - пулы.<br>
</details>

## Apache Airflow
<details>
<summary>Click to expand</summary>
      Clear task для перезапуска провальной таски.<br> для перезапуска сервера и счедулера - lsof -i tcp:port и затем kill PID.<br>
</details>

## Pytest (lab5)
<details>
<summary>Click to expand</summary>    
1. Проведено тестирование в JupiterNotebook данных на зашумление с использованием z-score.<br>              
2. Применены команды создания фаилов (%%writefile”имя файла”) из ячеек и запуска bash (! “имя команды”) из JupiterNotebook.<br>    
</details>

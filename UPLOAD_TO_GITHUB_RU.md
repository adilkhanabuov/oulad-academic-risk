# Как опубликовать репозиторий на GitHub

## Вариант через браузер

1. Войдите в GitHub и нажмите **New repository**.
2. Укажите имя:

   ```text
   oulad-course-conditioned-risk-prediction
   ```

3. Выберите **Public**.
4. Не добавляйте новый README, `.gitignore` или лицензию: они уже находятся в архиве.
5. Создайте репозиторий.
6. На странице пустого репозитория выберите **uploading an existing file** или **Add file → Upload files**.
7. Распакуйте ZIP и перетащите в окно GitHub всё содержимое папки `oulad-course-conditioned-risk-prediction`.
8. В поле сообщения коммита напишите:

   ```text
   Initial reproducibility release
   ```

9. Нажмите **Commit changes**.
10. Откройте `CITATION.cff` и замените `USERNAME` на имя вашего аккаунта GitHub.

## Ссылка для статьи

После публикации ссылка будет выглядеть так:

```text
https://github.com/USERNAME/oulad-course-conditioned-risk-prediction
```

В статью можно вставить формулировку:

> Скрипты подготовки данных, обучения, абляции и бутстреп-анализа доступны в открытом репозитории: (https://github.com/adilkhanabuov/oulad-academic-risk)

## Важно перед публикацией

- Не загружайте CSV-файлы OULAD из `data/oulad/raw/`.
- Замените `USERNAME` в `CITATION.cff` и в ссылке для статьи.
- При необходимости выберите лицензию через GitHub: **Add file → Create new file → Choose a license template**.


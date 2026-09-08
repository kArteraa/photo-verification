# Тестовые изображения

Все файлы взяты с Wikimedia Commons и опубликованы авторами под лицензией
CC0 1.0 Universal (public domain dedication). Для тестов исходники обрезаны
вокруг лица и уменьшены до высоты 512 px (пейзаж — до ширины 512 px).

| Файл | Источник | Автор | Лицензия |
|---|---|---|---|
| `portrait_frontal.jpg` | [File:Face portrait (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:Face_portrait_(Unsplash).jpg) | William Stitt | CC0 1.0 |
| `portrait_second.jpg` | [File:Thomas Hafeneth 2017-04-22 (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:Thomas_Hafeneth_2017-04-22_(Unsplash).jpg) | Thomas Hafeneth | CC0 1.0 |
| `portrait_turned.jpg` | [File:Sad face of a Wayuu Woman.jpg](https://commons.wikimedia.org/wiki/File:Sad_face_of_a_Wayuu_Woman.jpg) | Wilfredor | CC0 1.0 |
| `landscape.jpg` | [File:Mountain lake (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:Mountain_lake_(Unsplash).jpg) | Aleksandra Boguslawska | CC0 1.0 |

Назначение:

- `portrait_frontal.jpg` — фронтальное лицо крупным планом, пёстрый фон;
- `portrait_second.jpg` — фронтальное лицо на однородном светлом фоне, проходит сценарий документного фото;
- `portrait_turned.jpg` — голова повёрнута примерно на 25° по yaw;
- `landscape.jpg` — изображение без лица.

Тесты на этих файлах проверяют инварианты (число лиц, знак углов при отражении,
падение резкости при размытии), а не точные значения.

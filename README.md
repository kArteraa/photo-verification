# PhotoCheck

Прототип метода проверки фотографий пользователя на соответствие требованиям
с формированием мотивированного заключения: «факты → правила → вербализация».
Анализаторы извлекают из изображения измеримые факты, движок правил применяет
декларативную YAML-спецификацию требований, а заключение формируется только из
структурированного отчёта, без доступа к изображению.

## Установка

Требуется Python 3.12 и [uv](https://docs.astral.sh/uv/).

```
uv sync --all-groups
uv run photocheck models download
```

Команда `models download` скачивает две модели MediaPipe (детектор точек лица и
сегментатор человека, около 4 МБ) в каталог `models/`.

Если путь к проекту содержит символы вне ASCII, консольная команда `photocheck`
может не находить пакет (особенность Python 3.12 на Windows). Тогда используйте
эквивалентную форму `uv run python -m app ...`.

## Проверка фотографии

```
uv run photocheck check tests/fixtures/portrait_second.jpg --spec app/specs/document_photo.yaml --json out.json
```

Команда печатает таблицу вердиктов по каждому требованию, мотивированное
заключение на русском языке и путь к JSON-отчёту. Код возврата: 0 — фотография
принята, 1 — отклонена, 2 — ошибка спецификации или аргументов, 3 — ошибка
выполнения (нет изображения или моделей).

Сценарий задаётся файлом спецификации, код один и тот же:

- `app/specs/document_photo.yaml` — документное фото, жёсткие требования;
- `app/specs/avatar.yaml` — аватар площадки, мягче пороги;
- `app/specs/aesthetic.yaml` — эстетические рекомендации, все требования
  рекомендательные, фотография никогда не отклоняется.

## Заключение через языковую модель

```
uv run photocheck check IMG --spec app/specs/document_photo.yaml --llm
```

С флагом `--llm` заключение формулирует модель `claude-sonnet-4-6` через
Anthropic API, ключ берётся из переменной окружения `ANTHROPIC_API_KEY`. Модель
получает только JSON-отчёт, изображение ей не передаётся. Ответ автоматически
сверяется с отчётом: при расхождении выполняется одна регенерация, затем
используется шаблонный генератор. Без ключа или с флагом `--mock` работает
офлайн-клиент: он воспроизводит ранее записанные ответы из `results/llm_cache/`,
а при их отсутствии синтезирует ответ шаблонным генератором.

## Архитектура

Реализация повторяет формальную модель статьи. Требование
`r_i = (μ_i, π_i, τ_i, κ_i, D_i)` описывается строкой YAML: измеритель μ_i
(`measurer`), предикат π_i с порогами τ_i (`predicate`), тип κ_i
(`kind: hard | soft`) и зависимости D_i (`depends_on`). Система реализует
отображение `F: (x, R) → (v, m, e)`:

| Стадия | Код | Результат |
|---|---|---|
| Извлечение фактов | `app/analyzers/` (реестр измерителей) | вектор измерений m |
| Движок правил | `app/core/engine.py`, `app/core/spec.py` | вердикты v ∈ {pass, fail, undefined}, отчёт JSON |
| Вербализация | `app/conclusion/` | заключение e: шаблон или LLM со сверкой |

Вердикт `undefined` (⊥) ставится, когда не выполнена зависимость или
измерение неприменимо; для жёсткого требования это равносильно отказу.
Заключение считается консистентным, если каждое упомянутое нарушение есть в v,
а каждое число есть в m; проверка реализована в `app/conclusion/verify.py`.

### Как добавить своё требование

1. Напишите измеритель: чистую функцию от массивов и тонкую обёртку,
   зарегистрированную под точечным именем с перечнем ключей.

   ```python
   from app.analyzers.common import no_face
   from app.analyzers.context import AnalysisContext
   from app.analyzers.registry import register
   from app.core.report import Measurement


   def mouth_openness(landmarks_px) -> float:
       upper, lower = landmarks_px[13], landmarks_px[14]
       left, right = landmarks_px[61], landmarks_px[291]
       return float(abs(lower[1] - upper[1]) / max(abs(right[0] - left[0]), 1e-6))


   @register("mouth.openness", ("openness",))
   def measure_mouth(ctx: AnalysisContext) -> Measurement:
       face = ctx.primary_face
       if face is None:
           return no_face("mouth.openness")
       return Measurement.of("mouth.openness", {"openness": mouth_openness(face.landmarks_px)})
   ```

   Импортируйте модуль в `app/analyzers/__init__.py`.

2. Добавьте требование в YAML-сценарий:

   ```yaml
   - id: mouth_closed
     title: "Рот закрыт"
     measurer: mouth.openness
     predicate: {op: le, value: 0.08}
     kind: hard
     depends_on: [single_face]
     reason_template: "относительное раскрытие рта {value:.2f} при допуске {tau}"
     fix_hint: "закройте рот, сохраняя нейтральное выражение лица"
   ```

Поддерживаются операторы `eq`, `ge`, `le`, `range`, `abs_le`; в шаблонах
доступны ключи измерителя, `value`, `tau`, `tau_<ключ>`, для `range` также
`lo`, `hi`, `side`, `delta`, а для знаковых величин подсказки `<ключ>_dir` и
`<ключ>_abs`.

## Эксперименты

Данные не хранятся в репозитории. Валидные портреты берутся из FFHQ
(Karras T., Laine S., Aila T. A Style-Based Generator Architecture for
Generative Adversarial Networks // CVPR 2019;
https://github.com/NVlabs/ffhq-dataset), лицензия CC BY-NC-SA 4.0, подмножество
из 500 изображений 512×512. Нарушения порождаются контролируемыми
аугментациями, по одному варианту на класс из каждого валидного снимка:
размытие, затемнение и пересвет, замена фона процедурной текстурой,
уменьшение и смещение лица на однородном холсте, коллаж из двух портретов.
Класс `non_frontal` отбирается из исходников по измеренному повороту головы
(файл `results/non_frontal_check.csv` содержит выборку для ручной проверки).
Фон всех синтетических вариантов предварительно заменяется на однородный
холст цвета исходного фона, поэтому класс `clean` действительно чистый.
Разбиение 30 % калибровка / 70 % тест выполняется по исходному снимку.

Полная последовательность (пути по умолчанию, seed 42):

```
uv run python experiments/download_valid.py --n 500 --out data/valid
uv run python -m experiments.make_dataset
uv run python -m experiments.calibrate --mode frr --max-frr 0.05
uv run python -m experiments.run_eval
uv run python -m experiments.run_vlm_baseline --limit 100 --mock
uv run python -m experiments.plots
```

Калибровка переписывает пороги в `app/specs/document_photo.yaml` и сохраняет
ROC-кривые в `results/calibration.json`. Требования без собственного класса
нарушений калибровка не трогает, а пишет для них перцентили на чистых снимках;
их пороги заданы явно: допуск по yaw 15° совпадает с определением класса
`non_frontal`, допуск по pitch 18° и минимум EAR 0.15 соответствуют 95-му и
5-му перцентилям чистых снимков, то есть той же политике FRR ≤ 5 % на
требование. Оценка пишет
`results/raw_predictions.csv` и `results/metrics.json`. VLM-бейзлайн без ключа
или с флагом `--mock` использует детерминированную заглушку, и все его цифры
помечаются как mock в `results/table2.csv` и на графиках; с ключом
`ANTHROPIC_API_KEY` та же команда без `--mock` обращается к модели, ответы
кэшируются в `results/llm_cache/`. Итог: `results/table2.csv` (метрики по
требованиям), `results/table2_integral.csv` (FAR, FRR, время, свойства
заключений) и четыре рисунка в `results/figures/` (JPEG, 300 dpi).
Консистентность и полнота заключений считаются по обязательным требованиям:
для предложенного метода утверждения о нарушениях берутся из итоговых строк
заключения, для VLM из тегов в его объяснении.

## Тесты и проверка кода

```
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

Тесты на реальных портретах (`tests/fixtures/`, лицензия CC0, см.
`tests/fixtures/README.md`) помечены маркером `mediapipe` и пропускаются, пока
не скачаны модели. Тесты с маркером `dataset` требуют каталог `data/valid/`.

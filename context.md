# Рабочий контекст

## 2026-10-03 — Точка входа сервиса, тесты эндпоинтов, отдельный этап CI

**Задача:** добавить файл запуска сервиса, покрыть тестами его и эндпоинты,
вынести проверку поднятого сервера в отдельный этап CI. Попутно вычистить
мёртвый код и подключить `scripts/tracking` к baseline.

**Решения пользователя:** замер покрытия не нужен; `plots.py` и `metrics.py`
подключаются к baseline, а не удаляются; `model/data/processing_dataset.py` и
`model/checkpoints/__init__.py` удаляются из репозитория amend'ом в старые
коммиты, без нового коммита на удаление.

**Что было не так:**

- Файла `service/main.py` не было вообще, хотя README предлагал
  `uvicorn service.main:app`. Поднимать было нечего.
- `service/api.py` грузил токенизатор и веса **на уровне импорта** (строки
  18–26 старой версии). Из-за этого модуль нельзя было импортировать в
  тестах, а сервер нельзя было поднять без обученного чекпоинта. Именно
  поэтому API не был покрыт тестами в предыдущей записи.
- `scripts/tracking/metrics.py` и `plots.py` не импортировались ни одним
  production-модулем: их читали только собственные тесты. По графу импортов
  это был мёртвый код.
- `scripts/baseline.py` был скриптом верхнего уровня без функций, поэтому его
  нельзя было вызвать из теста — тест дублировал пайплайн вместо вызова.
- `scripts/error_analysis.py` логировал текст писем (два цикла по 5 строк),
  хотя README утверждает «текст писем не логируется, даже в error_analysis.py».
- Артефакты baseline (`baseline_*.csv`, `*.png`) не были в `.gitignore` и
  попали бы в коммит.

**Что сделано:**

- `service/api.py` — переписан: модель собирается в `build_runtime()` и
  загружается в `lifespan`, состояние лежит в `app.state`, эндпоинты берут
  его через `get_runtime(request)`. `setup_logging()` убран из импорта и
  переехал в точку входа.
- `service/main.py` (создан) — `main()` поднимает uvicorn с приложением из
  `service.api`, настраивает логирование. `python -m service.main` и
  `uvicorn service.main:app` работают.
- `service/stubs.py` (создан) — `TinyTokenizer`, `TinyEncoder`,
  `build_tiny_model()`, `fake_model_enabled()`. Включаются переменной
  окружения `INBOX_CLEANER_FAKE_MODEL=1`: сервер поднимается в CI без весов
  rubert (~700 МБ) и без сети. Значения разбираются строго: `"0"` и `"false"`
  заглушки **не** включают.
- `model/spam_classifier.py` — добавлен аргумент `encoder=None`. Точка
  подмены энкодера нужна и заглушкам, и тестам; без неё пришлось бы патчить
  `AutoModel.from_pretrained` глобально.
- `scripts/baseline.py` — переписан из скрипта верхнего уровня в функции
  `load_splits`, `build_model`, `run_baseline`, `save_artifacts`, `main`.
  Теперь считает метрики через `compute_all`, пишет `baseline_metrics.csv`
  и `baseline_predictions.csv` и строит график через `plot_test_comparison`.
  `tracking/` перестал быть мёртвым.
- `scripts/error_analysis.py` — удалены два цикла, печатавшие текст писем в
  лог: код расходился с правилом в README.
- `.gitignore` — добавлены `baseline_*.csv`, `baseline_*.png`,
  `training_curves.png`, `test_comparison.png`.
- `tests/test_api.py` (создан) — 15 тестов через `TestClient`: `/health`,
  `/predict`, границы `text` и `threshold`, 404, 405, детерминизм, строгость
  разбора флага заглушек.
- `tests/test_main.py` (создан) — 7 тестов: `main()` вызывает `uvicorn.run`
  с нужными host/port/reload, приложение импортируется без загрузки весов,
  lifespan настроен.
- `tests/test_baseline_pipeline.py` — переписан на вызов настоящих функций
  `scripts/baseline.py` (было: продублированный в тесте пайплайн).
- `.github/workflows/ci.yml` — добавлен второй джоб `api` с `needs: test`:
  поднимает `uvicorn service.main:app` на заглушках, ждёт готовности до
  30 секунд, проверяет `/health`, `/predict` и 422 на пустом тексте. Лог
  сервера пишется в `/tmp/uvicorn.log` и показывается при падении.

**Проверка:** `ruff check .` — `All checks passed!`; `pytest -q` —
`133 passed, 5 warnings`. Сервер поднят вручную теми же командами, что в CI:
`/health` вернул `{"status":"ok","device":"cpu","model_loaded":true}`,
`/predict` вернул `{"label":"ham","confidence":0.4369...}`, пустой текст дал
422.

**Ограничения и незакрытое:**

- Заглушки дают правдоподобный, но не осмысленный `confidence`: энкодер
  считается на эмбеддингах символов. Для проверки формы ответа и контракта
  этого достаточно, реальное качество проверяется обученной моделью.
- `service/api.py` по-прежнему читает пути весов константами
  (`model/checkpoints/best.pt`), а не настройками: в бою это работает, но
  путь не переопределяется без правки кода.
- Не тронуто, потому что выходит за подтверждённый объём работ: в
  зависимостях объявлены `pydantic-settings`, `python-dotenv`, `joblib` и
  `streamlit`, но не используются нигде; `.env.example` описывает настройки,
  которые никто не читает; README и `pyproject.toml` ссылаются на
  несуществующий слой `interface/` (в `packages` он тоже указан, хотя папки
  нет).

## 2026-10-03 — CPU-only torch в uv.lock

**Задача:** прогон CI на ubuntu-latest упирался в многогигабайтную загрузку
CUDA-зависимостей torch.

**Что было не так:** `uv lock` на linux резолвил `torch 2.14.1` из PyPI,
который тянет `cuda-toolkit` с 12 экстрами (cublas, cudnn, cufft, curand,
cusolver, cusparse, nccl, nvjitlink, nvrtc, nvtx) плюс `triton`. Это
несколько гигабайт на каждый прогон, и риск упереться в место на диске
раннера. Локально на Windows вставало `2.14.1+cpu`, поэтому проблема была
не видна: Windows-колесо всегда CPU, CUDA-вариант там не существует.

**Решение (выбрано пользователем):** зафиксировать CPU-колесо в локе.
Осознанное следствие — GPU-обучение локально перестаёт работать из-под `uv
sync`; torch для GPU ставится отдельно, инструкция в README.

**Что сделано:**

- `pyproject.toml` — `[tool.uv]` с явным индексом
  `https://download.pytorch.org/whl/cpu` и `explicit = true`, плюс
  `[tool.uv.sources]` с `torch = { index = "pytorch-cpu" }`. Индекс
  `explicit = true` обязателен: иначе torch искался бы ещё и в PyPI, и
  резолвился бы в CUDA-вариант.
- `uv.lock` перегенерирован: 20 CUDA-пакетов удалено (`cuda-toolkit`,
  `cuda-bindings`, `nvidia-*`, `triton` и др.), в локе ноль упоминаний
  cuda/nvidia/triton. Остались два блока torch, оба с CPU-индекса:
  `2.14.1` для darwin (на macOS нет суффикса `+cpu`) и `2.14.1+cpu` для
  win32 и linux.
- `README.md` — раздел «Torch: CPU по умолчанию» с объяснением причины и
  командой возврата GPU.

**Проверка:** `uv sync --dev` поставил `torch 2.14.1+cpu`;
`torch.cuda.is_available()` = `False`; `uv run ruff check .` —
`All checks passed!`; `uv run pytest -q` — `106 passed, 4 warnings`.

**Ограничение:** `uv sync` в uv 0.10.9 не умеет `--torch-backend` (он есть
только у `uv pip install`), поэтому принудительно CPU нельзя задать одной
командой в workflow — отсюда и правка индекса в `pyproject.toml`.

## 2026-10-03 — Тесты на код обучения, torch в зависимостях

**Задача:** покрыть автотестами независимые функции, в том числе код
обучения, и добиться, чтобы CI не пушил дальше при падении.

**Решения:** тестируем чистые функции вокруг обучения, а не сам цикл
обучения. `train_one_epoch`, early stopping, AMP и подбор `lr` юнит-тестами
не покрываются: проверяются запуском `python -m model.train`, а ассерты на
качество модели зависят от seed и версии железа. В CI ничего не скачивается
из HuggingFace — `from_pretrained` подменяется заглушкой через `monkeypatch`.

**Что было не так:**

- `torch` и `transformers` не были прописаны ни в `pyproject.toml`, ни в
  `uv.lock`, хотя `model/train.py`, `model/spam_classifier.py`,
  `model/data/dataset.py`, `model/tokenization/tokenizer.py` и `service/api.py`
  их импортируют — проект не запускался после `uv sync`.
- `scripts/tracking/plots.py` не существовал: все попытки его создания
  в предыдущей сессии молча провалились, тесты на графики падали с
  `ModuleNotFoundError`. Файл создан заново.
- `scripts/error_analysis.py`, `scripts/tracking/*` и `tests/test_metrics.py`
  были удалены из рабочей копии, хотя есть в коммите `70b2552`. Восстановлены
  через `git restore`; локальный `pytest` снова даёт полный набор.
- `ruff check .` падал на четырёх `B905` в `scripts/tracking/metrics.py`.
  Так как CI запускает `ruff`, пайплайн был бы красным.

**Что сделано:**

- `pyproject.toml` — добавлены `torch>=2.5` и `transformers>=4.46` в
  `[project] dependencies` (это рантайм, а не только тесты);
  `matplotlib` и `seaborn` были задвоены в `dependencies` и `dev` — оставлены
  в `dependencies`, из `dev` убраны.
- `scripts/tracking/metrics.py` — `zip(preds, targets, strict=True)`.
  Разная длина списков теперь ошибка, а не молчаливое усечение: потерянные
  примеры тихо портили бы метрики.
- `model/data/labels.py` — `frame[column].astype(str)` перед `.str.strip()`.
  На пустом фрейме pandas выводит тип `float64`, и `.str` падал с
  `AttributeError` вместо внятной ошибки про неизвестные метки.
  Заодно `decode_labels` получил докстринг.
- `scripts/tracking/plots.py` (создан) — `plot_training_curves` и
  `plot_test_comparison`. loss рисуется на левой оси, метрики — на правой:
  их масштабы не совпадают. Модуль не импортирует torch, поэтому строит
  графики и для baseline, и для RuBERT.
- `tests/conftest.py` (создан) — `FakeTokenizer`, `FakeModel`, `make_batch`,
  фикстуры `fake_tokenizer`, `tmp_csv`, `spam_ham_rows`. Здесь же
  `sys.path.insert` для корня репозитория, из тестовых файлов он убран.
- `tests/test_metrics.py` — 7 тестов, включая расхождение длин.
- `tests/test_preprocessing.py` — 17 тестов на `clean_text` по фактическому
  поведению: регистр и пунктуация сохраняются, удаляются только ссылки,
  кавычки, звёздочки и лишние пробелы.
- `tests/test_labels.py` — 13 тестов: регистр, пробелы, свой столбец, dtype,
  неизвестные метки, пустой фрейм, round-trip.
- `tests/test_schemas.py` — 8 тестов на `EmailRequest` (границы `text` и
  `threshold`), `SpamResponse`, `HealthResponse`.
- `tests/test_plots.py` — 5 тестов: PNG создан и непустой, вложенные каталоги
  создаются, пустой словарь метрик → `ValueError`. Бэкенд `Agg`, чтобы не
  открывать окно.
- `tests/test_logging_setup.py` — 10 тестов: нет дублей обработчиков, `INFO`
  только у пакета `model`, длинное сообщение заменяется заглушкой, чужие
  обработчики не трогаются.
- `tests/test_baseline_pipeline.py` — 7 тестов на TF-IDF+LR: пайплайн доходит
  до f1 = 1.0 на разделяемых словах, confusion сходится с числом строк.
- `tests/test_tokenizer.py` — 6 тестов на `TextTokenizer`: `max_length`,
  `truncation`, `padding`, батчи, `vocab_size`, запрет сетевого вызова.
- `tests/test_dataset.py` — 11 тестов на `SpamDataset`: `len`, ключи
  `__getitem__`, dtype `float32`, кэш пишется и читается повторно без
  повторной токенизации, `clean_text` применяется до токенизации, lazy-режим.
- `tests/test_classifier.py` — 9 тестов на `SpamClassifier`: форма `(B,)` после
  `squeeze(-1)`, градиенты текут и в энкодер, и в голову, детерминизм в
  `eval()`.
- `tests/test_evaluate.py` — 14 тестов на `evaluate()`: арифметика f1/precision/
  recall против ручного подсчёта, работа порога, склейка батчей.
- `.github/workflows/ci.yml` — `setup-uv@v5` с `enable-cache`,
  кэш `~/.cache/matplotlib` (иначе первый `import` пересобирает шрифты).
- `.github/workflows/github-actions-demo.yml` — удалён. Это был демо-шаблон
  из первого коммита GitHub, он срабатывал на каждом пуше и ничего не проверял.

**Проверка:** `uv run ruff check .` — `All checks passed!`;
`uv run pytest -q` — `106 passed, 4 warnings` за 9 секунд. Четыре
`UndefinedMetricWarning` ожидаемы: они возникают в тестах, где нет ни одного
положительного предсказания, и показывают реальное поведение sklearn.
YAML workflow проверен через `yaml.safe_load`.

**Ограничение:** веса rubert (~700 МБ) в тестах не участвуют — энкодер
заменён модулем с `nn.Embedding(100, 16)`. Поэтому тесты не ловят расхождение
реального rubert с ожидаемыми, а проверяют контракты кода: формы, dtype,
поток градиентов. `service/api.py` целиком не покрыт — поднятие FastAPI и
загрузка весов в CI дороже, чем польза; вместо этого покрыты `schemas.py`.
Качество модели (f1 выше порога) не проверяется намеренно.

## 2026-10-02 — Русский язык в коммитах, force-push dev

**Задача:** сообщения об ошибках на русском (сами ошибки библиотек английские),
все коммиты репозитория на русском, целостность коммитов и репозитория
сохранить.

**Что было в истории:** два английских сообщения — `ee25135 init: ML pipeline
architecture skeleton` и корневой `6ad1957 Initial commit`; остальные 22 уже
были на русском. Оба английских коммита уже опубликованы на origin, поэтому
перевод требует переписывания истории.

**Что сделано:**

- `common/logging_setup.py` — см. запись ниже (русские сообщения,
  `errors="replace"` вместо ASCII-экранирования).
- `model/train.py` — сообщения возвращены на русский, `Early stopping.` →
  `Ранняя остановка.`, оставлен `logger.setLevel(logging.INFO)`.
- История: `git filter-branch --msg-filter` по локальным веткам,
  `6ad1957` → `chore: создан репозиторий проекта inbox-cleaner`,
  `ee25135` → `init: разделение проекта на слои data/model/service/interface`;
  тела остальных коммитов не тронуты, авторы и даты сохранены.
- Перед пушем проверено: `git ls-remote` (только `dev` и `main`, ни тегов, ни
  `refs/pull/*`), `git fsck` без ошибок, dry-run `--force-with-lease` дал
  `forced update` без отказов.
- `git push --force-with-lease origin dev` выполнен: `origin/dev` = `aa5232c`
  (24 коммита), ветка совпадает с локальной, расхождения нет.
- `main` не переписывалась: продакшен получает содержимое из `dev` через запрос
  на слияние. Локально `main` = `d3775a2` (это `origin/main` плюс один
  непушнутый коммит, форс не нужен). Переписанный вариант `main` сохранён
  локально в ветке `main-ru` (`f917070`) на случай, если понадобится.
- Резервные ссылки `refs/original/refs/heads/dev` (`0a4a534`) и
  `refs/original/refs/heads/main` (`d3775a2`) оставлены для отката.

**Последствия:** у клонов, созданных до пуша, `dev` разошёлся с локальной
веткой, там нужен `git fetch && git reset --hard origin/dev`. Пушить рабочие
изменения логирования в `dev` пока нельзя — они не закоммичены намеренно.

**Ограничение:** переведённые сообщения попали на remote только для `dev`;
корневой коммит в `main` остался `Initial commit`, и перевести его можно лишь
форсом продакшен-ветки.

## 2026-10-02 — INFO для обучения, русские сообщения

**Задача:** INFO должен быть виден в разделах обучения модели, сообщения об
ошибках — на русском (сами ошибки библиотек остаются английскими), текст писем
в терминал не попадает ни при каких условиях.

**Решения:** INFO поднят всему пакету `model` (выбран весь пакет, не только
`train.py`). Пробовали ASCII-экранирование сообщений — откатил: русский текст
превращался в `\uXXXX`, а требовалось его читать. Вместо этого кодировка потока
не меняется, а на нём ставится `errors="replace"`.

**Что сделано:**

- `common/logging_setup.py`
  - `INFO_PACKAGES = ("model",)` — `setup_logging()` поднимает этим логгерам
    `INFO`, остальные остаются на `WARNING`;
  - `MAX_MESSAGE_CHARS = 1000` и фильтр `SuppressLongMessages`: длинное
    сообщение заменяется заглушкой «длинное сообщение скрыто (N символов)»,
    поэтому текст письма в консоль не выводится даже при ошибке в коде;
  - удалён `AsciiFormatter`, добавлен `_safe_stream()`: кодировку потока не
    меняем (в консоли это cp1251/cp866, от utf-8 был бы мусор), только
    `errors="replace"` — символ вне кодировки печатается как `?` вместо
    `UnicodeEncodeError`;
  - правила дополнены: свои сообщения на русском, чужие исключения не переводим.
- `model/train.py`
  - `logger.setLevel(logging.INFO)` с комментарием: `python -m model.train`
    выполняет файл как `__main__`, и пакетная настройка его не покрывает;
  - `Early stopping.` → `Ранняя остановка.`, остальные сообщения остались
    русскими, перевод на английский откачен;
  - добавлен `import logging`.
- `model/data/dataset.py`, `service/api.py`, `scripts/baseline.py`,
  `scripts/error_analysis.py` — сообщения вернулись на русский, правок нет.
- `README.md` — раздел «Логирование»: русские сообщения, `errors="replace"`,
  `INFO` только у пакета `model`, заглушка вместо длинных сообщений.

**Проверка:** `uv run ruff check common model service scripts` — чисто (кроме
не моих `B905` в `scripts/tracking/metrics.py`). Smoke-скрипт: два вызова
`setup_logging()` дают один обработчик; `model.train` и `model.data.dataset`
печатают `INFO`, `service.api` — нет; `warning` печатается; письмо на 1088
символов заменяется заглушкой; эмодзи печатается как `?` без исключения;
после настройки `sys.stderr.errors` равен `replace`.

**Ограничение:** фильтр ловит длинные тексты; короткое письмо, если его явно
передать в `logger.info`, отфильтровать нельзя — это правило ревью. В пайпе
агента кириллица выглядит мусором из-за cp1251, в интерактивной консоли
Windows текст читается.

## 2026-10-02 — Черновик: ASCII-сообщения (откачено)

**Задача:** кириллица не выводится в консоль, INFO только для обучения.

**Почему откачено:** сообщения `model/train.py` были переведены на английский,
а `AsciiFormatter` экранировал всё не-ASCII. INFO обучения нужен читаемым, а
`\u041e\u0431\u0443\u0447...` нечитаемо; замена — `errors="replace"` вместо
экранирования, см. запись выше.

**Ограничение:** фильтр ловит длинные тексты; короткое письмо, если его явно
передать в `logger.info`, отфильтровать нельзя — это правило ревью.

## 2026-10-02 — Минимальное логирование

**Задача:** в `common` оставить самое простое логирование — без файлов, без
кириллицы в терминале, в строке только модуль и текст, порог `WARNING`.

**Что сделано:**

- `common/logging_setup.py` (переписан целиком)
  - формат `DEFAULT_FORMAT = "%(name)s: %(message)s"` — только имя модуля и текст,
    без времени и уровня; `DEFAULT_LEVEL = logging.WARNING`, `INFO` не выводится;
  - добавлен `AsciiFormatter`: результат `format()` кодируется в ascii с
    `backslashreplace`, поэтому кириллица и эмодзи печатаются как `\uXXXX`
    и cp1251-консоль не падает;
  - обработчик помечается именем `HANDLER_NAME = "inbox-cleaner"`, повторный
    `setup_logging()` удаляет и закрывает только его — дубликатов нет, чужие
    обработчики (uvicorn, pytest) не трогаются;
  - удалены `LEVEL_ENV_VAR`, `DEFAULT_DATE_FORMAT`, `NOISY_LOGGERS`,
    `NOISY_LEVEL`, `_resolve_level()` и чтение переменной окружения; на пороге
    `WARNING` приглушать сторонние логгеры не нужно;
  - файловых обработчиков нет, вывод только в stderr.
- `README.md` — раздел «Логирование» переписан под новые правила, убраны строка
  про `INBOX_CLEANER_LOG_LEVEL` и блок «Подробный лог обучения».
- 40 вызовов `logger.info` в `model/`, `service/`, `scripts/` оставлены как есть:
  они больше не видны в терминале, поэтому прогресс обучения и метрики нужно
  либо поднять до `warning`, либо печатать через `print`.

**Проверка:** `uv run ruff check common/logging_setup.py` — чисто; smoke-вызов
`setup_logging()` дважды подряд оставляет один обработчик, `info`/`debug` не
печатаются, кириллица и эмодзи экранируются. Тестов в репозитории нет,
`uv run pytest` падает на отсутствии testpaths.

## 2026-09-30 — Упрощение логирования

**Задача:** консоль Windows по умолчанию в cp1251 и падает на символах из писем.
`_utf8_stream` переводил поток в utf-8 с заменой символов, но вывод текста
писем в лог не планируется, поэтому хак с перекодировкой не нужен.

**Что сделано:**

- `common/logging_setup.py`
  - удалён хелпер `_utf8_stream` и перекодировка `sys.stderr`;
  - `setup_logging()` потерял параметры `fmt` и `noisy_level` — формат и уровень
    приглушения берутся из констант `DEFAULT_FORMAT` и новая `NOISY_LEVEL`;
  - обработчик создаётся на `sys.stderr if stream is None else stream`;
  - проверка `noisy_level is not None` заменена на `resolved > logging.DEBUG`,
    так как параметра больше нет.
- `scripts/error_analysis.py`
  - удалён вывод текста писем FP/FN в лог (два цикла по 5 примеров);
  - удалён комментарий про кодировку вывода — он ссылался на `_utf8_stream`;
  - сами CSV с примерами (`error_analysis_fp.csv`, `error_analysis_fn.csv`)
    сохраняются как раньше, текст писем там остаётся.
- `README.md` — в таблицу правил логирования добавлена строка «Текст писем не
  логируется» с причиной (cp1251).

**Проверка:** `uv run ruff check common/logging_setup.py scripts/error_analysis.py` —
всё чисто; smoke-вызов `setup_logging()` дважды подряд не дублирует обработчики.
Тестов в репозитории нет, `uv run pytest` падает на отсутствии testpaths.

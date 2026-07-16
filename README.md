# Рекомендации банковских продуктов на следующий месяц

Финальный проект решает задачу Santander Product Recommendation: по профилю и
текущему портфелю клиента ранжирует банковские продукты, которые он может
подключить в следующем календарном месяце. Последний доступный снимок датирован
28 мая 2016 года, поэтому продуктовый прогноз финального артефакта относится к
июню 2016 года.

В проекте использованы Python 3.10/3.11, pandas, scikit-learn, MLflow, FastAPI,
Pydantic, Docker Compose, Prometheus и Grafana. Репозиторий содержит полный цикл:
EDA → временная разметка → эксперименты → MLflow → бинарный артефакт → API →
мониторинг.

## Результат

Модель проверяется на последнем полностью размеченном переходе апрель→май 2016
года. В качестве простого ориентира используется популярность новых покупок.
С ней сравниваются линейная модель и несколько вариантов смешивания её оценок с
популярностью.

Оценка проводится по всем клиентам проверочного периода, в том числе по тем, у
кого в следующем месяце не появилось новых продуктов. Поэтому небольшие
абсолютные значения метрик ожидаемы для настолько разреженной задачи.

Модель можно обучить прямо в `notebooks/02_modeling.ipynb` или командой
`bash scripts/train.sh`. Оба варианта используют полный набор данных и сохраняют
сравнение моделей в
[`reports/model_comparison.json`](reports/model_comparison.json), а параметры
выбранной модели — в
[`reports/model_metadata.json`](reports/model_metadata.json).

## Бизнес-задача и её техническая трансляция

Бизнесу нужен короткий персональный список продуктов для следующего контакта с
клиентом. Чем выше нужный продукт в списке, тем меньше нерелевантных предложений
и затрат на коммуникацию. Offline-прокси бизнес-эффекта — качество top-7:

- основная метрика — `MAP@7`: учитывает и наличие покупки, и позицию продукта;
- `Recall@7` — доля фактических новых покупок, попавших в выдачу;
- `Precision@7` — доля релевантных позиций в семи рекомендациях;
- `Coverage@7` — доля 24-продуктового каталога, появляющаяся в выдачах;
- macro PR-AUC — способность ранжировать клиентов внутри каждого редкого label.

Продукт считается новой покупкой только при положительном изменении между
строго соседними календарными месяцами одного клиента:

```text
X(t) = профиль клиента и его текущие продукты в месяце t
y(t, product) = max(product(t + 1) - product(t), 0)
```

Снятие продукта `1→0` не является покупкой. Пара после пропущенного месяца или
при неизвестном состоянии хотя бы одного из 24 продуктов не строится. При выдаче
уже подключённые продукты получают маску и не могут быть рекомендованы.

Случайное разделение строк здесь создало бы утечку одного клиента и будущих
месяцев. Обучение и подбор выполняются на переходах до марта→апрель 2016, а
единственный holdout — апрель→май. После фиксации решения финальная модель
переобучается на всех парах до апреля→май и применяется к майскому снимку. Метрик
за июнь нет: июньских фактических покупок в исходном файле не предоставлено.

## Архитектура

```mermaid
flowchart LR
    A["train_ver2.csv<br/>13.6 млн строк"] --> B["Один pandas.read_csv<br/>все клиенты в памяти"]
    B --> C["Соседние месяцы<br/>target 0→1"]
    C --> D["Temporal holdout<br/>апрель→май 2016"]
    D --> E["Feature engineering<br/>impute + scale + OHE"]
    E --> F["24 One-vs-Rest<br/>SGD classifiers"]
    F --> G["MLflow tracking server<br/>external PostgreSQL + S3"]
    F --> H["ml_models/model.joblib"]
    H --> I["FastAPI<br/>/predict + /batch"]
    I --> J["Prometheus"]
    J --> K["Grafana dashboard"]
```

Сервис загружает модель один раз при старте. Обучение внутри API не выполняется.
Если файл отсутствует или нарушает контракт, процесс остаётся доступным для
диагностики, а `/health` и prediction endpoints возвращают `503`.

## Данные и EDA

`train_ver2.csv` содержит 13 647 309 строк, 48 колонок, 956 645 клиентов и 17
ежемесячных снимков с `2015-01-28` по `2016-05-28`. Первые 24 поля описывают
клиента, последние 24 — наличие продуктов. Сводка полного набора зафиксирована в
[`reports/eda_summary.json`](reports/eda_summary.json), анализ — в
[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

Ключевые выводы:

- число строк в месяце выросло с 625 457 до 931 453; наиболее заметный скачок —
  между июнем и июлем 2015 (632 110 → 829 817, или +31.3%);
- 605 464 клиента наблюдаются все 17 месяцев, у 8 018 клиентов есть разрывы
  внутри их истории, дубликатов `клиент–месяц` нет;
- паспорт данных охватывает все 48 полей, включая `fecha_alta`,
  `ult_fec_cli_1t` и `conyuemp`; последние два пусты соответственно в 99.82% и
  99.99% строк, а у `fecha_alta` найдено 11 586 дат позже снимка;
- среди последних профилей есть 7 490 возрастов младше 18 лет, 992 старше
  100 лет и три отрицательных значения стажа; у `renta` 20.48% пропусков и
  длинный правый хвост;
- `indrel_1mes` и `cod_prov` приведены к единому строковому представлению,
  `tipodom` фактически константен, а среди 162 каналов привлечения 125 встречаются
  реже чем в 0.1% последних профилей;
- распространённость продуктов сильно различается: текущий счёт встречается
  в 8 945 588 клиентских месяцах, а банковская гарантия — всего в 316; в
  финальном месяце 25.22% клиентов не имеют продуктов, 48.63% имеют один;
- для всех 24 продуктов отдельно считаются исходные пропуски, владение по
  клиентским месяцам, уникальные владельцы, состояние последнего снимка,
  подключения `0→1` и отключения `1→0`;
- неизвестное состояние продукта не считается отсутствием при EDA переходов:
  пары с пропуском исключаются из статистики конкретного продукта, а в обучение
  попадают только соседние месяцы со всеми известными продуктовыми флагами;
- из 12 669 584 полностью известных соседних пар только 3.52% содержат хотя бы
  одно подключение; при попродуктовом подсчёте по всем допустимым для конкретного
  флага парам чаще всего подключается прямое дебетование — 153 146 событий;
- для трёх продуктов в holdout апрель→май нет покупок;
- активные клиенты совершают хотя бы одно подключение в 68.54 случаях на 1000
  пар против 6.73 у неактивных; покупки разрежены и меняются во времени, поэтому
  accuracy и случайный split непригодны.

Исходный CSV весит около 2.29 GB. EDA загружает все 48 колонок одним вызовом
`pandas.read_csv` и использует компактные типы `int32`, `float32`, `Int8` и
`category`. Это сохраняет исходные пропуски и позволяет анализировать всех
клиентов без нескольких широких копий таблицы. Перед повторным запуском
ноутбука рекомендуется перезапустить ядро, чтобы освободить память.

## Признаки и модель

В pipeline входят:

- числовые поля `age`, `antiguedad`, `renta`, клиентские статусы и 24 флага
  текущего портфеля;
- категориальные поля пола, страны, сегмента, канала, провинции и статусов;
- `snapshot_year`, `snapshot_month`, число текущих продуктов,
  `tenure_years`, `log_income`;
- median imputation с индикаторами пропусков и `StandardScaler` для чисел;
- constant imputation и `OneHotEncoder(handle_unknown="infrequent_if_exist")`
  для категорий;
- 24 независимых `SGDClassifier(loss="log_loss")` через One-vs-Rest.

Предобработка обучается только на train. У неизвестной категории на inference
нет ошибки. Финальная выдача сортируется детерминированно и дополнительно
проверяется сервисом на дубликаты и уже подключённые продукты.

Ноутбук `02_modeling.ipynb` и команда `bash scripts/train.sh` загружают полный
CSV, обучают модель на всех доступных клиентах и сохраняют готовый артефакт в
[`ml_models/model.joblib`](ml_models/model.joblib). Вместе с моделью заново
создаются отчёты с параметрами и метриками.

## Структура репозитория

```text
.
├── notebooks/
│   ├── 01_eda.ipynb                 # исследование данных и покупки 0→1
│   └── 02_modeling.ipynb            # baseline, модель, temporal holdout
├── src/bank_recommender/
│   ├── constants.py                 # единый список 24 продуктов и признаки
│   ├── data.py                      # полная загрузка CSV и temporal target
│   ├── features.py                  # feature engineering и preprocessing
│   ├── metrics.py                   # MAP/precision/recall/coverage/PR-AUC
│   ├── model.py                     # baseline и сериализуемая модель
│   └── train.py                     # обучение, refit, MLflow, export
├── ml_models/model.joblib           # готовый бинарный артефакт
├── reports/                         # EDA, сравнение и model metadata, JSON
├── examples/predict_request.json    # пример запроса
├── scripts/                         # MLflow, train, API, smoke test
├── services/
│   ├── docker-compose.yaml          # API, MLflow, Prometheus и Grafana
│   ├── recomendation_service/
│   │   ├── Dockerfile               # минимальный non-root API image
│   │   └── app/
│   │       ├── schemas.py           # Pydantic-схемы
│   │       ├── predictor.py         # загрузка и safety checks
│   │       └── main.py              # FastAPI и Prometheus-метрики
│   ├── mlflow/                      # image tracking server
│   ├── prometheus/                  # scrape config и alert rules
│   └── grafana/                     # datasource и dashboard provisioning
├── tests/                           # unit + API tests
├── Monitoring.md                    # метрики, PromQL и runbook
├── .env.example                     # шаблон внешнего PostgreSQL и S3
├── requirements.txt                 # полное исследовательское окружение
└── requirements-runtime.txt         # минимальное окружение API image
```

CSV, parquet, `.cache`, `.env` и произвольные model dumps исключены через
`.gitignore`. Готовый `ml_models/model.joblib` добавлен явным
исключением и остаётся в репозитории.

## Установка и воспроизведение окружения

```bash
git clone https://github.com/nike173145/mle-pr-final.git
cd mle-pr-final
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Для PowerShell активация выглядит так:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Проект проверен с Python 3.10; Docker image использует Python 3.11. Поместите
полученный из задания `train_ver2.csv` в корень. Файл намеренно не хранится в
Git.

Все случайные состояния модели (`random`, NumPy и SGD) используют seed 42.
Путь к данным, MLflow URI и путь к модели задаются аргументами или переменными
окружения.

## Ноутбуки

Из корня проекта:

```bash
jupyter lab
```

Оба ноутбука загружают полный CSV в память одним чтением и обрабатывают всех
клиентов. `01_eda.ipynb` читает все 48 исходных колонок и уже сохранён с
результатами полного выполнения: 25 кодовых ячеек завершились без ошибок.
`02_modeling.ipynb` использует `read_snapshots(DATA_PATH)`, обучает два варианта
SGD с разной регуляризацией, сравнивает их с popularity baseline и сохраняет
лучший вариант SGD. Его результаты очищены, поэтому для просмотра экспериментов
ноутбук нужно выполнить заново. Перед запуском убедитесь, что машине достаточно
RAM.

Проверка чистого последовательного исполнения:

```bash
mkdir -p .tmp
jupyter nbconvert --execute --to notebook notebooks/01_eda.ipynb \
  --output 01_eda.executed.ipynb --output-dir .tmp \
  --ExecutePreprocessor.timeout=-1
jupyter nbconvert --execute --to notebook notebooks/02_modeling.ipynb \
  --output 02_modeling.executed.ipynb --output-dir .tmp \
  --ExecutePreprocessor.timeout=-1
```

## MLflow и обучение

MLflow намеренно использует только внешние хранилища: PostgreSQL для metadata и
сторонний S3/S3-compatible storage для артефактов. В Compose нет локальных
PostgreSQL, MinIO, SQLite backend или volume с MLflow-артефактами. До запуска:

```bash
cp .env.example .env
```

Заполните `.env` значениями своего провайдера:

```dotenv
MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://mlflow:p%40ss%3Aword@db.example:5432/mlflow?sslmode=require
MLFLOW_ARTIFACTS_DESTINATION=s3://existing-mlflow-bucket/bank-recommender
MLFLOW_S3_ENDPOINT_URL=https://s3.example.com
AWS_DEFAULT_REGION=ru-central1
AWS_ACCESS_KEY_ID=replace-me
AWS_SECRET_ACCESS_KEY=replace-me
```


Проверка конфигурации и запуск tracking server без Docker:

```bash
bash scripts/run_mlflow.sh --check
bash scripts/run_mlflow.sh
```

Рекомендуемый запуск всего контура упакован в shell-скрипт:

```bash
bash scripts/run_stack.sh
```

Эквивалентные команды Compose для запуска только MLflow и проверки статуса:

```bash
docker compose --env-file .env -f services/docker-compose.yaml up --build -d mlflow
docker compose --env-file .env -f services/docker-compose.yaml ps
```

Сервер стартует с `--serve-artifacts`: training client отправляет артефакты на
tracking URI, а MLflow проксирует их в S3. Поэтому клиенту обучения достаточно
`MLFLOW_TRACKING_URI`; AWS-ключи нужны контейнеру MLflow, а не ноутбуку или
процессу обучения. UI доступен на `http://127.0.0.1:5000`.

Запуск обучения и регистрации модели:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
bash scripts/train.sh --registered-model-name bank-product-recommender
```

С тем же `MLFLOW_TRACKING_URI` можно выполнить все ячейки
`notebooks/02_modeling.ipynb`. Ноутбук запишет в один run параметры выбранной
модели и метрики всех сравнённых вариантов. Run также содержит JSON-отчёты,
sklearn model, input example и signature. Финальный локальный артефакт сохраняется
в `ml_models/model.joblib`. Обучение можно воспроизвести и без регистрации:

```bash
python -m bank_recommender.train \
  --data train_ver2.csv \
  --seed 42 \
  --artifact ml_models/model.joblib
```

Обе команды всегда загружают и обрабатывают полный набор клиентов, поэтому для
запуска потребуется достаточно оперативной памяти и времени.

## API

Локальный запуск с готовым артефактом:

```bash
bash scripts/run_api.sh
```

Без shell-скрипта:

```bash
export MODEL_PATH=ml_models/model.joblib
uvicorn services.recomendation_service.app.main:app --host 0.0.0.0 --port 8000
```

Документация OpenAPI доступна на `http://127.0.0.1:8000/docs`.

| Method | Endpoint | Назначение |
|---|---|---|
| GET | `/health` | готовность и версия загруженной модели |
| POST | `/predict` | рекомендации для одного клиента |
| POST | `/predict/batch` | до 100 профилей за запрос |
| GET | `/metrics` | Prometheus exposition format |

Обязательные поля single request: `fecha_dato`, `age`, `antiguedad`, `renta`.
Остальные профильные поля опциональны. `current_products` содержит уникальные
имена из 24 продуктовых колонок, `top_k` находится в диапазоне 1…24. Полная схема
и пример находятся в
[`examples/predict_request.json`](examples/predict_request.json).

```bash
curl --fail -X POST \
  -H 'Content-Type: application/json' \
  --data @examples/predict_request.json \
  http://127.0.0.1:8000/predict
```

Фрагмент ответа:

```json
{
  "customer_id": 1375586,
  "model_version": "sgd-ovr-v1",
  "recommendations": [
    {"product": "ind_recibo_ult1", "score": 0.01284953, "rank": 1},
    {"product": "ind_cno_fin_ult1", "score": 0.00259519, "rank": 2}
  ]
}
```

`score` используется для относительного ранжирования и не является обещанной
вероятностью покупки или ожидаемым финансовым эффектом. Ошибка схемы даёт `422`,
отсутствующий/повреждённый артефакт или ошибка инференса — `503`.

Batch body имеет вид `{"requests": [<single request>, ...]}`. Для совместимости
принимаются также ключ `items` и raw JSON list.

## Docker и наблюдаемость

```bash
cp .env.example .env
# заполните внешний PostgreSQL и S3 в .env
docker compose --env-file .env -f services/docker-compose.yaml config
docker compose --env-file .env -f services/docker-compose.yaml up --build -d
docker compose --env-file .env -f services/docker-compose.yaml ps
bash scripts/smoke_test.sh
```

Логи и остановка контура:

```bash
docker compose --env-file .env -f services/docker-compose.yaml logs -f --tail=100
docker compose --env-file .env -f services/docker-compose.yaml down
```

После запуска доступны:

- API и Swagger — `http://127.0.0.1:8000/docs`;
- MLflow — `http://127.0.0.1:5000`;
- Prometheus — `http://127.0.0.1:9090`;
- Grafana — `http://127.0.0.1:3000`.

API image запускается non-root, имеет healthcheck, read-only filesystem,
отключённые Linux capabilities и отдельный минимальный requirements-файл.
MLflow хранит metadata во внешнем PostgreSQL и проксирует артефакты во внешний
S3. Локальных PostgreSQL/MinIO, SQLite и MLflow data volumes нет. Именованные
volumes используются только Prometheus и Grafana; datasource и dashboard
provisioned автоматически из `services/grafana`.

Все опубликованные порты привязаны к `127.0.0.1`. MLflow и Grafana в этом
учебном контуре не имеют production-аутентификации, поэтому их нельзя открывать
в публичную сеть. `.env` исключён из Git; используйте отдельного PostgreSQL
пользователя, минимальные S3-права на один prefix, TLS, ротацию ключей и secret
manager оркестратора вместо передачи секретов в образ или репозиторий.

Сервис экспортирует:

- request count/status и latency;
- prediction/failure/empty-output counters;
- число и частоту рекомендованных продуктов;
- распределение ranking score;
- число отфильтрованных уже имеющихся продуктов;
- histogram возраста и дохода для сигналов входного drift;
- стандартные process CPU/RAM metrics.

PromQL, alert rules, ограничения labels и порядок реакции описаны в
[`Monitoring.md`](Monitoring.md).

## Проверки

```bash
python -m pip check
python -m compileall -q src services tests
pytest -q
```

Тесты проверяют target `0→1`, удаление продукта, пропущенный месяц, дубликаты,
полную загрузку снимков, точность ranking metrics, фильтрацию текущего портфеля,
serialization round-trip, API-схемы, batch, `422`, `503`, health и `/metrics`.
CI выполняет те же unit- и API-тесты на Python 3.11.

Перед публикацией:

```bash
git diff --check
git ls-files | grep -E 'train_ver2|mle-project-sprint-[234]|(^|/)\.env$' && exit 1 || true
git grep -nE 'AWS_SECRET_ACCESS_KEY|SECRET_KEY\s*=|PASSWORD\s*=' || true
```

## Ограничения и развитие

- Полное обучение требует заметного объёма оперативной памяти и времени. Для
  крайне редких продуктов положительных событий может оказаться недостаточно,
  поэтому их качество нужно отслеживать отдельно.
- Holdout только один: для более строгой оценки нужны rolling backtests по
  нескольким месяцам и доверительные интервалы.
- Offline MAP не измеряет конверсию, маржу, стоимость контакта и риск продукта.
  Нужны бизнес-правила, propensity/uplift-постановка и контролируемый A/B-тест.
- Score линейной модели не откалиброван как бизнес-вероятность. Для пороговых
  решений нужна temporal calibration на отдельном периоде.
- Monitoring наблюдает proxy drift по агрегатам. Для формального контроля нужны
  PSI/KS/JS по обезличенным срезам и отложенный расчёт MAP@7 после появления
  фактических покупок.
- API принимает полный профиль и не подключён к feature store; в production
  идентификатор клиента должен разрешаться в версионированный online feature
  snapshot с контролем свежести и доступа к персональным данным.

## Соответствие заданию

| Требование | Артефакт |
|---|---|
| EDA | `notebooks/01_eda.ipynb`, `reports/eda_summary.json` |
| MLflow + внешний PostgreSQL/S3 | `scripts/run_mlflow.sh`, `services/mlflow/`, `.env.example`, логирование в `src/bank_recommender/train.py` |
| Трансляция и метрики | разделы выше и `src/bank_recommender/metrics.py` |
| Эксперименты + bin-модель | `notebooks/02_modeling.ipynb`, `reports/`, `ml_models/model.joblib` |
| API + Docker | `services/recomendation_service/`, `examples/` |
| Compose + мониторинг | `services/docker-compose.yaml`, `services/prometheus/`, `services/grafana/`, `Monitoring.md` |
| Мониторинг из кода | `services/recomendation_service/app/main.py`, `Monitoring.md` |
| Документация | этот `README.md` |
| Среда и seeds | pinned `requirements*.txt`, seed 42, `.env.example` |

# Мониторинг сервиса банковских рекомендаций

## Назначение и архитектура

Контур мониторинга позволяет отделить проблемы инфраструктуры от ошибок API,
деградации модели и изменений входных данных.
`services/docker-compose.yaml` поднимает четыре сервиса в общей сети:

- `api` — FastAPI-приложение на внутреннем порту `8000`;
- `prometheus` — сбор `/metrics`, хранение временных рядов за 15 дней и вычисление alert rules;
- `grafana` — автоматически настроенный read-only дашборд;
- `mlflow` — tracking server с внешним PostgreSQL backend и сторонним
  S3/S3-compatible artifact store.

Наружу порты публикуются только на loopback-интерфейсе. Их можно переопределить
через переменные из `.env`, созданного на основе `.env.example`.

| Компонент | Адрес по умолчанию | Проверка готовности |
|---|---|---|
| API | `http://localhost:8000` | `GET /health` |
| Prometheus | `http://localhost:9090` | `GET /-/healthy` |
| Grafana | `http://localhost:3000` | `GET /api/health` |
| MLflow | `http://localhost:5000` | `GET /health` |

MLflow получает URI внешней базы из `MLFLOW_BACKEND_STORE_URI`, а S3-prefix —
из `MLFLOW_ARTIFACTS_DESTINATION`. Сервер запущен с `--serve-artifacts`, поэтому
клиент обучения отправляет файлы на MLflow HTTP endpoint, а сервер проксирует их
в S3. AWS-ключи не нужны клиенту обучения и должны находиться только в окружении
MLflow. В Compose нет локальных PostgreSQL, MinIO, SQLite backend, volume с
MLflow metadata или volume с артефактами.

Служебные файлы MLflow Prometheus exporter эфемерны и размещены в `tmpfs`, чтобы
после перезапуска не оставались устаревшие worker-серии. Только данные Prometheus
и Grafana вынесены в именованные volumes и сохраняются при пересоздании
контейнеров. Prometheus собирает собственные метрики, метрики API, MLflow и
Grafana; Docker healthchecks независимо контролируют готовность контейнеров.

## Запуск

До сборки должен существовать `ml_models/model.joblib`. Создайте `.env` и
заполните подключение к внешним сервисам:

```bash
cp .env.example .env
```

Минимальная конфигурация выглядит так:

```dotenv
MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://mlflow:p%40ss%3Aword@db.example:5432/mlflow?sslmode=require
MLFLOW_ARTIFACTS_DESTINATION=s3://existing-mlflow-bucket/bank-recommender
MLFLOW_S3_ENDPOINT_URL=https://s3.example.com
AWS_DEFAULT_REGION=ru-central1
AWS_ACCESS_KEY_ID=replace-me
AWS_SECRET_ACCESS_KEY=replace-me
```

База, пользователь PostgreSQL и S3 bucket должны существовать до запуска.
Пароль в PostgreSQL URI кодируется как URL (`@` → `%40`, `:` → `%3A`,
`/` → `%2F`), а удалённое соединение должно использовать
`?sslmode=require`. S3-ключу нужны минимальные права на чтение, запись,
перечисление и удаление объектов только внутри выбранного prefix. Для временных
ключей добавьте `AWS_SESSION_TOKEN`, для частного CA — `AWS_CA_BUNDLE`.

Проверка внешних хранилищ и запуск всего контура через shell-скрипт:

```bash
bash scripts/run_mlflow.sh --check
bash scripts/run_stack.sh
```

Эквивалентный ручной запуск и диагностика Compose:

```bash
docker compose --env-file .env -f services/docker-compose.yaml config
docker compose --env-file .env -f services/docker-compose.yaml up --build -d
docker compose --env-file .env -f services/docker-compose.yaml ps
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/metrics
```

Логи всех сервисов:

```bash
docker compose --env-file .env -f services/docker-compose.yaml logs -f --tail=100
```

Остановка без удаления накопленных данных:

```bash
bash scripts/stop_stack.sh
```

Именованные volumes удаляются только с `down --volumes`; это очищает локальную
историю Prometheus и состояние Grafana, но не удаляет metadata MLflow из внешнего
PostgreSQL и не удаляет S3-артефакты. Управляйте retention и удалением внешних
данных средствами соответствующего провайдера.

## Слои метрик

### Инфраструктура

Стандартный Python collector публикует процессные метрики:

- `process_resident_memory_bytes` — реально занятая процессом память;
- `process_virtual_memory_bytes` — размер виртуальной памяти;
- `process_cpu_seconds_total` — накопленное CPU-время;
- `process_open_fds` — число открытых файловых дескрипторов на Linux;
- `process_start_time_seconds` — позволяет заметить перезапуск процесса.

Рост resident memory без возврата к обычному уровню может означать утечку или увеличение размера входных batch. Скачок CPU вместе с ростом latency чаще указывает на перегрузку инференса.

### HTTP и доступность

- `bank_api_requests_total{method,endpoint,status_code}` (`Counter`) — число ответов API;
- `bank_api_request_duration_seconds{method,endpoint}` (`Histogram`) — длительность запросов;
- `up{job="bank-api"}` — успешность scrape; это техническая доступность `/metrics`, а не проверка качества модели.

Labels должны иметь ограниченное множество значений. В `endpoint` следует записывать шаблон маршрута, например `/predict`, а не исходный URL. Идентификаторы клиентов и текст ошибок в labels добавлять нельзя: это создаёт высокую кардинальность и может раскрыть персональные данные.

### Модель и бизнес-выход

- `bank_predictions_total` (`Counter`) — число обработанных моделью объектов;
- `bank_recommendations_total{product}` (`Counter`) — число выданных рекомендаций по каждому продукту;
- `bank_recommendation_score` (`Histogram`) — распределение ranking score;
- `bank_empty_recommendations_total` (`Counter`) — число профилей с пустой выдачей;
- `bank_owned_products_filtered_total` (`Counter`) — число уже подключённых продуктов, исключённых из кандидатов;
- `bank_prediction_failures_total{reason}` (`Counter`) — ошибки инференса с ограниченным набором причин.

Разница между количеством запросов и предсказаний помогает обнаружить ошибки валидации или инференса. Отношение прироста `bank_recommendations_total` к `bank_predictions_total` показывает среднюю длину выдачи. Резкое падение может означать, что модель не находит кандидатов или фильтр уже подключённых продуктов исключает почти всё.

Эти online-метрики не заменяют `MAP@7`, `Recall@7` и остальные offline-метрики: фактические покупки становятся известны с задержкой, поэтому качество следует периодически пересчитывать на размеченном временном срезе и логировать в MLflow.

Offline target — именно новая покупка в следующем строго соседнем календарном
месяце: `y(t, product) = max(product(t + 1) - product(t), 0)`. Снятие продукта
`1→0` и текущее владение не являются положительным label. При пересчёте качества
нужно сохранять temporal split, считать `MAP@7` основной метрикой и дополнять её
`Precision@7`, `Recall@7`, `Coverage@7` и macro PR-AUC.

### Входные данные и дрейф

- `bank_input_age` (`Histogram`) — распределение возраста во входных запросах;
- `bank_input_income` (`Histogram`) — распределение дохода.

Возраст и доход выбраны как понятные числовые индикаторы population drift. Дашборд показывает их скользящие p50 и p95. Изменение этих квантилей само по себе не доказывает деградацию: оно является сигналом проверить источник данных, доли пропусков, сегменты и offline-качество.

Prometheus histogram хранит агрегированные buckets, поэтому для формального контроля дрейфа необходим отдельный периодический расчёт PSI/KS/Jensen–Shannon divergence по обезличенным срезам. Сравнивать нужно с зафиксированным training baseline, а пороги утверждать после наблюдения нормальной недельной и месячной сезонности.

## Полезные PromQL-запросы

Общая интенсивность запросов:

```promql
sum(rate(bank_api_requests_total[5m]))
```

Доля ответов 5xx:

```promql
(sum(rate(bank_api_requests_total{status_code=~"5.."}[5m])) or vector(0))
/
clamp_min(sum(rate(bank_api_requests_total[5m])), 0.001)
```

Задержка p95:

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(bank_api_request_duration_seconds_bucket{endpoint="/predict"}[5m])
  )
)
```

Число предсказаний за час:

```promql
sum(increase(bank_predictions_total[1h]))
```

Среднее число рекомендованных продуктов на предсказание:

```promql
sum(increase(bank_recommendations_total[1h]))
/
clamp_min(sum(increase(bank_predictions_total[1h])), 1)
```

Квантили возраста и дохода за 30 минут:

```promql
histogram_quantile(0.50, sum by (le) (rate(bank_input_age_bucket[30m])))
histogram_quantile(0.95, sum by (le) (rate(bank_input_income_bucket[30m])))
```

Память и CPU процесса:

```promql
process_resident_memory_bytes{job="bank-api"}
100 * sum(rate(process_cpu_seconds_total{job="bank-api"}[5m]))
```

## Дашборд Grafana

Datasource и dashboard загружаются автоматически из `services/grafana`. Ручная
настройка и передача паролей не требуются: локальный Grafana работает в
анонимном режиме Viewer без формы входа и без возможности менять provisioned
dashboard.

Dashboard `Bank product API overview` содержит панели:

- request rate;
- p50/p95 latency;
- интенсивность предсказаний и число рекомендованных продуктов;
- доступность target и доля 5xx;
- p50/p95 возраста и дохода;
- resident/virtual memory;
- CPU процесса.

Если панель пуста сразу после запуска, нужно сделать несколько запросов к `/predict`: для `rate()` и `histogram_quantile()` требуются как минимум две точки scrape и наблюдения в соответствующих histogram.

## Алерты и реакция

Правила находятся в `services/prometheus/alerts.yml` и автоматически загружаются
Prometheus.

| Alert | Условие | Первичная реакция |
|---|---|---|
| `BankApiUnavailable` | `up == 0` более 2 минут | Проверить Compose status, healthcheck, логи API и наличие модели |
| `BankApiHighErrorRate` | доля 5xx выше 5% в течение 10 минут | Разбить ошибки по endpoint, проверить traceback, модель и схему входа |
| `BankApiHighP95Latency` | p95 `/predict` выше 1 секунды 10 минут | Проверить CPU/RAM, размер batch и время preprocessing/inference |
| `ObservabilityServiceUnavailable` | MLflow или Grafana не scrape-ится 5 минут | Проверить healthcheck, сеть compose и логи соответствующего сервиса |
| `BankApiHighMemoryUsage` | RSS выше 1 GiB 15 минут | Сравнить размер модели и batch, искать накопление объектов |
| `BankApiPredictionsStopped` | запросы есть, предсказаний нет | Проверить 4xx/5xx, загрузку модели и формат запросов |
| `BankApiSuspiciousAgeP95` | p95 возраста выше 100 | Проверить единицы, парсинг, выбросы и изменение источника данных |

Пороговые значения являются стартовыми и должны быть откалиброваны на реальном
профиле нагрузки. Prometheus показывает состояние alerts, но в Compose намеренно
не включён Alertmanager. Для production необходимо подключить Alertmanager и
маршрутизацию уведомлений.

Порты API, MLflow, Prometheus и Grafana привязаны к `127.0.0.1`. Tracking UI и
анонимный Grafana нельзя публиковать в интернет без reverse proxy,
аутентификации и TLS. `.env` исключён из Git: храните его только локально, не
передавайте секреты в Docker image и используйте secret manager в оркестраторе.
Для PostgreSQL заведите отдельного пользователя, для S3 ограничьте ключ одним
bucket/prefix и регулярно ротируйте credentials. Не выводите URI, AWS-ключи,
персональные входы API или тексты ошибок в Prometheus labels и логи.

## Проверка контура после изменений

1. `docker compose --env-file .env -f services/docker-compose.yaml config` — проверить итоговую конфигурацию и подстановку портов; не прикладывать вывод с секретами к публичному issue.
2. `docker compose --env-file .env -f services/docker-compose.yaml up --build -d` и `docker compose --env-file .env -f services/docker-compose.yaml ps` — все четыре сервиса должны стать healthy.
3. В Prometheus открыть `Status → Targets`: jobs `prometheus`, `bank-api`, `mlflow` и `grafana` должны иметь состояние `UP`.
4. Отправить валидные и ошибочные запросы к API; убедиться, что counters и histograms меняются.
5. Проверить появление dashboard без ручного импорта.
6. Остановить и снова поднять stack без `--volumes`; история Prometheus должна
   сохраниться локально, а история MLflow — во внешних PostgreSQL/S3 независимо
   от жизненного цикла контейнера.

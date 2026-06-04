# BSAC Portfolio Management

## Структура проекта

```
└── 📁 bsac_portfolio
    └── 📁 app
        ├── 📁 modules                # [6]
        ├── app.py                    # [6]
        ├── market_loader.py          # [1]
    └── 📁 db_init
        ├── db_scheme.sql
        ├── podman-compose.yml
        ├── upsert.py
    └── 📁 models
        ├── 📁 aggressive_v1
        ├── 📁 conservative_v1
        └── 📁 moderate_v1
    └── 📁 src
        └── 📁 bsac_portfolio
            ├── 📁 bsac               # [4]
            ├── 📁 data_preparation   # [3]
            ├── 📁 database           # [2]
            ├── 📁 market_client      # [1]
            └── 📁 portfolio          # [5]
    └── config.toml
```
#### Обозначения
| № | Подсистема                                        |
| - | -------------------------------- |
| [1] | Подсистема сбора и синхронизации рыночных данных |
| [2] | Подсистема хранения и управления данными |
| [3] | Подсистема предобработки данных и генерации признаков |
| [4] | Подсистема обучения и инференса моделей глубокого обучения с подкреплением |
| [5] | Подсистема управления портфелем |
| [6] | Подсистема пользовательского интерфейса и управления сессиями |

## Требования

| Компонент                                 | Назначение                       |
| ----------------------------------------- | -------------------------------- |
| **Git**                                   | Клонирование репозитория         |
| **Python 3.10**                           | Основной runtime проекта         |
| **Poetry**                                | Управление зависимостями         |
| **Podman** *(рекомендуется)* / **Docker** | Контейнерный runtime             |
| **podman-compose** / **docker compose**   | Запуск и управление контейнерами |

## Развёртывание приложения

### 1. Клонирование проекта

Клонирование с помощью Git:

```bash
git clone https://github.com/dmlapochkin/bsac_portfolio.git
```

Переход в папку с проектом:

```bash
cd bsac_portfolio
```

### 2. Установка зависимостей

> *Из корня  проекта*

Poetry может быть установлен через pip:

```bash
pip install poetry
```

Настройка окружения Poetry на Python 3.10:

```bash
poetry env use python3.10   # Linux/MacOS
poetry env use py -3.10     # Windows
```

#### Установка зависимостей:

```bash
poetry install
```

### 3. Настройка конфигурации

#### `db_init/podman-compose.yml`

Укажите параметры контейнера PostgreSQL:

```yaml
    ...
    environment:
      POSTGRES_USER: USER           # ← имя пользователя для доступа к БД
      POSTGRES_PASSWORD: PASSWORD   # ← пароль для доступа к БД
    ...
```

#### `app/config.toml`

Параметры подключения должны совпадать с `podman-compose.yml`:

```toml
[database]
user = "USER"               # ← USER совпадает с POSTGRES_USER
password = "PASSWORD"       # ← USER совпадает с POSTGRES_PASSWORD

...

[application]
mode = "live"               # режим: "live" или "demo"
demo_date = "2021-02-01"    # дата начала демо-режима (mode = "demo")
...
```

### 4. Запуск базы данных

Переход в папку для инициализации:

```bash
cd db_init
```

Запуск БД в контейнере:

```bash
podman-compose up -d
```
> `podman-compose.yml` совместим с `docker compose`.

Проверка статуса контейнера:

```bash
podman ps   # или docker ps
```
Инициализация таблиц:

> USER должен совпадать с POSTGRES_USER

```bash
cat db_scheme.sql | podman exec -i bsac-portfolio \
      psql -U USER -d bsac-portfolio-db
```

### 5. Загрузка данных

> *Из корня  проекта*

`upsert.py` заполняет справочники:
- securities — список инструментов MOEX
- indices — список индексов
- risk_profiles — профили риска (Агрессивный, Умеренный, Консервативный)

```bash
poetry run python -m db_init.upsert
```
Рыночные данные загружаются через `market_loader.py`:

```bash
poetry run python -m app.market_loader
```

## Запуск веб-приложения приложения

> *Из корня  проекта*

```bash
poetry run streamlit run app/app.py
```
> Приложение открывается на `http://localhost:8501`

## Использование планировщика обновлений

Документация:
```bash
poetry run python -m app.market_loader --help
```

### Ручное обновление

Обновить все данные:
```bash
poetry run python -m app.market_loader
```

Обновить только рыночные данные:
```bash
poetry run python -m app.market_loader --market
```

Обновить только данные индекса:
```bash
poetry run python -m app.market_loader --index
```

Обновить только состав индекса:
```bash
poetry run python -m app.market_loader --composition
```

Загрузка за указанный период:
```bash
poetry run python -m app.market_loader --start 2026-01-01 --end 2026-02-01
```

Указать конкретные тикеры или индексы:
```bash
poetry run python -m app.market_loader --tickers SBER,GAZP
poetry run python -m app.market_loader --indexes IMOEX,RGBI
```

### Автоматическое обновление

Запуск в фоновом режиме:
```bash
poetry run python -m app.market_loader --auto
```

> Автоматическое обновление выполняется  после закрытия основной сессии MOEX (после 19:00 МСК в торговые дни).
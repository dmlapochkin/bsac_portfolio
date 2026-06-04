"""
Модуль загрузки и обновления рыночных данных через MOEX ISS.

Поддерживает два режима работы:
1. Ручная загрузка данных за указанный период
2. Автоматическое обновление по расписанию после закрытия торгов (19:00 МСК)
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import time
import threading
from typing import Optional, Callable

import toml
import click

from bsac_portfolio.market_client.market_client import MarketDataClient
from bsac_portfolio.database.db_manager import DatabaseManager


SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config.toml"
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path(__file__).parent.parent / "config.toml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    APP_CONFIG = toml.load(f)

db_cfg = APP_CONFIG.get("database", {})
DSN = f"postgresql://{db_cfg.get('user', '')}:{db_cfg.get('password', '')}@{db_cfg.get('host', 'localhost')}:{db_cfg.get('port', 5432)}/{db_cfg.get('database', '')}"

DEFAULT_TICKERS = APP_CONFIG.get("market_data", {}).get("default_tickers", [])
TARGET_INDEX = APP_CONFIG.get("market_data", {}).get("target_indices", ["IMOEX"])[0]
MSK_TZ = ZoneInfo("Europe/Moscow")

db = DatabaseManager(dsn=DSN)
client = MarketDataClient()


def _get_effective_date():
    """
    Возвращает последнюю доступную торговую дату с учётом времени закрытия торгов.

    Если сейчас будний день и время после 19:00 МСК, возвращает сегодняшнюю дату.

    Returns:
        Дата последнего доступного торгового дня
    """
    now = datetime.now(MSK_TZ)
    today = now.date()
    
    if now.weekday() < 5 and now.hour >= 19:
        return today
    
    end = today - timedelta(days=1)
    while end.weekday() >= 5:
        end -= timedelta(days=1)
    return end


def _is_after_market_close():
    """
    Проверяет, наступило ли время после закрытия торгов MOEX (19:00 МСК).

    Returns:
        True, если сейчас будний день и время после 19:00 МСК, иначе False
    """
    now = datetime.now(MSK_TZ)
    return now.weekday() < 5 and now.hour >= 19


def update_market_data(start_date: date = None, end_date: date = None, db_manager: DatabaseManager = None, tickers: list[str] = None):
    """
    Обновляет данные market_data для всех доступных тикеров.
    
    Args:
        start_date: Начальная дата загрузки (по умолчанию - последняя дата в БД)
        end_date: Конечная дата загрузки (по умолчанию - последняя доступная торговая дата)
        db_manager: Экземпляр DatabaseManager (по умолчанию используется глобальный db)
        tickers: Список тикеров для загрузки (приоритет над данными из БД и config.toml)
    
    Returns:
        Количество добавленных записей
    """
    print("Обновление market_data")
    db_mgr = db_manager if db_manager else db
    
    if tickers is not None:
        print(f"Используются указанные тикеры: {', '.join(tickers)}")
    else:
        tickers = db_mgr.get_available_tickers()
        if not tickers:
            if DEFAULT_TICKERS:
                tickers = DEFAULT_TICKERS
                print(f"Тикеры в market_data не найдены. Используются дефолтные: {', '.join(tickers)}")
            else:
                print("Нет тикеров. Сначала загрузите справочник или задайте default_tickers в config.toml.")
                return 0
    
    if start_date is None:
        if tickers is not None:
            start = date(2020, 1, 1)
        else:
            start = db_mgr._get_last_db_date("market_data")
    else:
        start = start_date
    
    if end_date is None:
        end = _get_effective_date()
    else:
        end = end_date
    
    if start >= end:
        print("market_data актуально")
        return 0
    
    total = 0
    for t in tickers:
        try:
            df = client.fetch_candles(t, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), interval=24)
            if not df.empty:
                total += db_mgr.upsert_market_data(df)
                print(f"[{t}] +{len(df)}")
        except Exception as e:
            print(f"[{t}] Ошибка: {e}")
    
    print(f"Добавлено: {total}")
    return total


def update_index_data(index_ids: list[str] = None, start_date: date = None, end_date: date = None, db_manager: DatabaseManager = None) -> int:
    """
    Обновляет данные индекса.
    
    Args:
        index_ids: Список ID индексов для обновления
        start_date: Начальная дата загрузки
        end_date: Конечная дата загрузки
        db_manager: Экземпляр DatabaseManager
    
    Returns:
        Количество добавленных записей
    """
    if index_ids is None:
        index_ids = [TARGET_INDEX]
    print(f"Обновление index_data ({', '.join(index_ids)})")
    db_mgr = db_manager if db_manager else db
    
    if start_date is None:
        start = db_mgr._get_last_db_date("index_data", "index_id", index_ids[0])
    else:
        start = start_date
    
    if end_date is None:
        end = _get_effective_date()
    else:
        end = end_date
    
    if start >= end:
        print("index_data актуально")
        return 0
    
    total = 0
    for index_id in index_ids:
        try:
            df = client.fetch_candles(index_id, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), interval=24, market="index")
            if not df.empty:
                df["index_id"] = index_id
                count = db_mgr.upsert_index_data(df)
                print(f"[{index_id}] +{count}")
                total += count
        except Exception as e:
            print(f"[{index_id}] Ошибка: {e}")

    return total


def update_index_composition(index_ids: list[str] = None, ref_date: date = None, start_date: date = None, end_date: date = None, db_manager: DatabaseManager = None) -> int:
    """Обновляет состав индекса.
    
    Args:
        index_ids: Список ID индексов
        ref_date: Дата для получения состава (по умолчанию - последняя доступная)
        start_date: Начальная дата диапазона для пакетной загрузки состава
        end_date: Конечная дата диапазона для пакетной загрузки состава
        db_manager: Экземпляр DatabaseManager
    
    Returns:
        Количество добавленных записей
    """
    if index_ids is None:
        index_ids = APP_CONFIG.get("market_data", {}).get("target_indices", ["IMOEX"])
    db_mgr = db_manager if db_manager else db
    
    if start_date and end_date:
        print(f"Обновление состава {', '.join(index_ids)} за период {start_date} — {end_date}...")
        total = 0
        for index_id in index_ids:
            current = start_date.replace(day=1)
            end_month = end_date.replace(day=1)
            while current <= end_month:
                try:
                    df = client.fetch_index_structure(index_id, current.strftime("%Y-%m-%d"))
                    if not df.empty:
                        count = db_mgr.upsert_index_composition(df)
                        print(f"[{index_id}] {current.strftime('%Y-%m')} +{count}")
                        total += count
                except Exception as e:
                    print(f"[{index_id}] {current.strftime('%Y-%m')} Ошибка: {e}")
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
        return total
    
    if ref_date is None:
        ref_date = _get_effective_date()
    print(f"Обновление состава {', '.join(index_ids)} на {ref_date}...")
    
    total = 0
    for index_id in index_ids:
        try:
            df = client.fetch_index_structure(index_id, ref_date.strftime("%Y-%m-%d"))
            if not df.empty:
                count = db_mgr.upsert_index_composition(df)
                print(f"{index_id} +{count}")
                total += count
        except Exception as e:
            print(f"{index_id} Ошибка: {e}")
    
    return total


def update_all_data(db_manager = None) -> None:
    """Выполняет полное обновление всех данных.
    
    Args:
        db_manager: Экземпляр DatabaseManager
    """
    update_market_data(db_manager=db_manager)
    update_index_data(db_manager=db_manager)
    update_index_composition(db_manager=db_manager)


class RealTimeUpdater:
    """
    Класс для автоматического обновления данных.
    
    Проверяет поступление новых данных после закрытия торгов MOEX (19:00 МСК).
    """
    
    def __init__(self, check_interval: int = 900, db_manager: DatabaseManager = None) -> None:
        """
        Инициализация планировщика обновлений.
        
        Args:
            check_interval: Интервал проверки в секундах (по умолчанию 15 минут)
            db_manager: Экземпляр DatabaseManager
        """
        self.check_interval = check_interval
        self._db_manager = db_manager if db_manager else db
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_update_date: Optional[date] = None
        self._on_update_callback: Optional[Callable] = None
    
    def set_on_update_callback(self, callback: Callable) -> None:
        """Устанавливает callback-функцию, вызываемую после успешного обновления."""
        self._on_update_callback = callback
    
    def _check_and_update(self) -> None:
        """Проверяет необходимость обновления и выполняет его."""
        if not _is_after_market_close():
            return
        
        today = date.today()
        
        if self._last_update_date == today:
            return
        
        last_market_date = self._db_manager._get_last_db_date("market_data")
        effective_date = _get_effective_date()
        
        if last_market_date >= effective_date:
            return
        
        print(f"\n[{datetime.now(MSK_TZ).strftime('%Y-%m-%d %H:%M:%S')}] Начало обновления")
        
        try:
            update_all_data(db_manager=self._db_manager)
            self._last_update_date = today
            print(f"[{datetime.now(MSK_TZ).strftime('%Y-%m-%d %H:%M:%S')}] Обновление завершено")
            
            if self._on_update_callback:
                self._on_update_callback()
        except Exception as e:
            print(f"[{datetime.now(MSK_TZ).strftime('%Y-%m-%d %H:%M:%S')}] Ошибка обновления: {e}")
    
    def _run_loop(self) -> None:
        while self._running:
            self._check_and_update()
            time.sleep(self.check_interval)
    
    def start(self) -> None:
        """Запускает фоновое обновление."""
        if self._running:
            print("Обновление уже запущено")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"Автоматическое обновление запущено (интервал: {self.check_interval} сек)")
    
    def stop(self):
        """Останавливает фоновое обновление."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        print("Автоматическое обновление остановлено")
    
    def is_running(self) -> bool:
        """Проверяет, запущено ли автоматическое обновление."""
        return self._running


real_time_updater = RealTimeUpdater()


@click.command()
@click.option("--all", "update_all", is_flag=True, help="Обновить все данные (по умолчанию)")
@click.option("--market", is_flag=True, help="Обновить только market_data")
@click.option("--index", is_flag=True, help="Обновить только index_data")
@click.option("--composition", is_flag=True, help="Обновить только состав индекса")
@click.option("--auto", is_flag=True, help="Запустить автоматическое обновление в реальном времени")
@click.option("--start", type=str, help="Начальная дата в формате YYYY-MM-DD")
@click.option("--end", type=str, help="Конечная дата в формате YYYY-MM-DD")
@click.option("--index-id", type=str, default=TARGET_INDEX, help=f"ID индекса (по умолчанию {TARGET_INDEX})")
@click.option("--interval", type=int, default=300, help="Интервал проверки в секундах (по умолчанию 300)")
@click.option("--tickers", multiple=True, help="Список тикеров через запятую (например, --tickers SBER,GAZP). Приоритет над данными из БД и config.toml")
@click.option("--indexes", multiple=True, help="Список индексов через запятую (например, --indexes IMOEX,RGBI). Приоритет над config.toml")
def main(update_all: bool, market: bool, index: bool, composition: bool, auto: bool, 
         start: str, end: str, index_id: str, interval: int,
         tickers: tuple[str], indexes: tuple[str]) -> None:
    """
    Обновление рыночных данных MOEX.
    
    Поддерживает ручную загрузку данных и автоматическое обновление
    по расписанию после закрытия торгов (19:00 МСК).
    
    Примеры использования:
        python -m app.modules.market_data_updater
        python -m app.modules.market_data_updater --market
        python -m app.modules.market_data_updater --index --index-id IMOEX
        python -m app.modules.market_data_updater --start 2026-01-01 --end 2026-02-01
        python -m app.modules.market_data_updater --auto --interval 900
    """
    start_date = None
    end_date = None
    
    if start:
        try:
            start_date = date.fromisoformat(start)
        except ValueError:
            click.echo(f"Ошибка: некорректный формат начальной даты '{start}'. Используйте YYYY-MM-DD")
            return
    
    if end:
        try:
            end_date = date.fromisoformat(end)
        except ValueError:
            click.echo(f"Ошибка: некорректный формат конечной даты '{end}'. Используйте YYYY-MM-DD")
            return
    
    if auto:
        click.echo(f"Запуск автоматического обновления (интервал: {interval} сек)")
        click.echo("Нажмите Ctrl+C для остановки")
        
        updater = RealTimeUpdater(check_interval=interval, db_manager=db)
        
        try:
            updater.start()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            click.echo("Остановка автоматического обновления")
            updater.stop()
            click.echo("Готово")
        return
    
    parsed_tickers: list[str] = []
    if tickers:
        for t in tickers:
            parsed_tickers.extend([x.strip() for x in t.split(",") if x.strip()])
    
    parsed_indexes: list[str] = []
    if indexes:
        for i in indexes:
            parsed_indexes.extend([x.strip() for x in i.split(",") if x.strip()])
    
    if market:
        update_market_data(start_date=start_date, end_date=end_date, db_manager=db, tickers=parsed_tickers or None)
    elif index:
        update_index_data(index_ids=parsed_indexes or [index_id], start_date=start_date, end_date=end_date, db_manager=db)
    elif composition:
        update_index_composition(index_ids=parsed_indexes or [index_id], start_date=start_date, end_date=end_date, db_manager=db)
    else:
        if start_date or end_date:
            update_market_data(start_date=start_date, end_date=end_date, db_manager=db, tickers=parsed_tickers or None)
            update_index_data(index_ids=parsed_indexes or [index_id], start_date=start_date, end_date=end_date, db_manager=db)
            update_index_composition(index_ids=parsed_indexes or [index_id], start_date=start_date, end_date=end_date, db_manager=db)
        else:
            update_all_data(db_manager=db)


if __name__ == "__main__":
    main()

import hashlib
import uuid
from datetime import date, timedelta
from typing import Optional

import colorsys
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras


class DatabaseManager:
    """Класс менеджера для работы с СУБД PostgreSQL."""

    def __init__(self, dsn: str = None):
        """
        Инициализация менеджера базы данных.

        Args:
            dsn: Строка подключения к базе данных.
        """
        self.dsn = dsn


    def _get_connection(self) -> psycopg2.extensions.connection:
        """
        Устанавливает новое подключение к базе данных.

        Returns:
            Объект подключения psycopg2.
        """
        return psycopg2.connect(self.dsn)


    def _clean_df(self, df: pd.DataFrame, cols: list[str]) -> list:
        """
        Очищает DataFrame от NaN значений и преобразует даты.

        Args:
            df: Исходный DataFrame.
            cols: Список колонок для очистки и преобразования.
        
        Returns:
            Список с очищенными данными.
        """
        df = df[cols].copy().replace({np.nan: None})
        for c in cols:
            if "date" in c.lower():
                df[c] = pd.to_datetime(df[c]).dt.date
        return df.values.tolist()


    def _color_from_hash(self, key: str) -> str:
        """
        Генерирует светлый цвет в формате HEX на основе хэша строки.

        Args:
            key: Входная строка для генерации цвета.

        Returns:
            Строка с HEX кодом цвета.
        """
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        hue = (h % 360) / 360.0
        sat = 0.65
        light = 0.55
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        r = min(255, max(0, int(r * 255)))
        g = min(255, max(0, int(g * 255)))
        b = min(255, max(0, int(b * 255)))
        return f'#{r:02x}{g:02x}{b:02x}'
    

    def _dark_color_from_hash(self, key: str) -> str:
        """
        Генерирует тёмный цвет в формате HEX на основе хэша строки.

        Args:
            key: Входная строка для генерации цвета.

        Returns:
            Строка с HEX кодом цвета.
        """
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        hue = (h % 360) / 360.0
        sat = 0.55 + (h % 50) / 100.0
        light = 0.30 + (h % 40) / 100.0
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        r = min(255, max(0, int(r * 255)))
        g = min(255, max(0, int(g * 255)))
        b = min(255, max(0, int(b * 255)))
        return f'#{r:02x}{g:02x}{b:02x}'


    def _execute_batch(self, query: str, data: list) -> int:
        """
        Универсальный метод пакетной вставки/обновления.
        
        Args:
            query: SQL запрос с плейсхолдером для данных (%s).
            data: Список данных для вставки/обновления.

        Returns:
            Количество обработанных строк.
        """
        if not data:
            return 0      
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    psycopg2.extras.execute_values(cur, query, data, page_size=5000)
                    conn.commit()
                    return len(data)
                except Exception as e:
                    conn.rollback()
                    raise


    def get_security_info(self, ticker: str) -> Optional[dict]:
        """
        Получает информацию об активе по тикеру.

        Args:
            ticker: Тикер актива.

        Returns:
            Словарь с информацией об активе.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM securities WHERE ticker = %s", (ticker,))
                return cur.fetchone()
            

    def get_available_tickers(self) -> list[str]:
        """
        Получает список доступных тикеров.

        Returns:
            Список тикеров.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT ticker FROM market_data ORDER BY ticker")
                rows = cur.fetchall()
        return [r[0] for r in rows] if rows else []


    def get_index_composition(self, date: str) -> Optional[pd.DataFrame]:
        """
        Получает состав индекса на заданную дату.

        Args:
            date: Дата в формате 'YYYY-MM-DD'.
        
        Returns:
            DataFrame с составом индекса.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM index_composition WHERE date = %s", (date, ))
                rows = cur.fetchall()
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame(rows)
                

    def get_index_data(self, index: str, start: str, end: str, warmup_days: int = 0) -> Optional[pd.DataFrame]:
        """
        Получает данные по индексу за указанный период.

        Args:
            index: id индекса.
            start: Начальная дата в формате 'YYYY-MM-DD'.
            end: Конечная дата в формате 'YYYY-MM-DD'.
            warmup_days: Количество дней для разминки (по умолчанию 0).
        
        Returns:
            DataFrame с данными по индексу.
        """
        start_dt = pd.to_datetime(start)
        if warmup_days > 0:
            actual_start = (start_dt - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        else:
            actual_start = start
        query = """
            SELECT date, index_id, open, high, low, close, volume, value
            FROM index_data
            WHERE index_id = %s AND date BETWEEN %s AND %s
            ORDER BY date ASC
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (index, actual_start, end))
                rows = cur.fetchall()  
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df


    def get_market_data(self, tickers: list[str], start: str, end: str, warmup_days: int = 0) -> Optional[pd.DataFrame]:
        """
        Получает данные рынка для заданных тикеров за указанный период.

        Args:
            tickers: Список тикеров.
            start: Начальная дата в формате 'YYYY-MM-DD'.
            end: Конечная дата в формате 'YYYY-MM-DD'.
            warmup_days: Количество дней для разминки (по умолчанию 0).

        Returns:
            DataFrame с рыночными данными.
        """
        start_dt = pd.to_datetime(start)
        if warmup_days > 0:
            actual_start = (start_dt - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        else:
            actual_start = start 
        query = """
            SELECT date, ticker, open, high, low, close, volume, value
            FROM market_data
            WHERE ticker = ANY(%s) AND date BETWEEN %s AND %s
            ORDER BY date ASC
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (tickers, actual_start, end))
                rows = cur.fetchall()  
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df

    
    def get_latest_market_data(self, tickers: list[str], target_date: date = None) -> Optional[dict]:
        """
        Получает последние данные рынка для заданных тикеров.

        Args:
            tickers: Список тикеров.
            target_date: Целевая дата для получения данных (по умолчанию - сегодня).

        Returns:
            Словарь с последними данными рынка для каждого тикера.
        """
        if target_date is not None:
            date_cond = "m.date = (SELECT MAX(date) FROM market_data WHERE date <= %s)"
            params = (tickers, target_date)
        else:
            date_cond = "m.date = (SELECT MAX(date) FROM market_data)"
            params = (tickers,)

        query = f"""
            SELECT s.ticker, s.lot_size, m.close, s.color
            FROM securities s
            JOIN market_data m ON s.ticker = m.ticker
            WHERE s.ticker = ANY(%s)
            AND {date_cond}
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                return {r['ticker']: {'close': float(r['close']), 'lot_size': int(r['lot_size']), 'color': r['color']} for r in cur.fetchall()}
            

    def upsert_indices(self, df: pd.DataFrame) -> int:
        """
        Вставляет информацию об индексах в базу данных.

        Args:
            df: DataFrame с информацией об индексах.
        
        Returns:
            Количество обработанных строк.
        """
        df = df.copy()
        df['index_id'] = df['SECID']
        df['name'] = df['NAME']
        df['short_name'] = df['SHORTNAME']
        df['board'] = df['BOARDID']
        df['currency'] = df['CURRENCYID']
        df["color"] = df["index_id"].apply(lambda x: self._dark_color_from_hash(x))
        cols = [
            'index_id', 'name', 'short_name', 'color',
            'board', 'currency'
        ]
        data = self._clean_df(df, cols)  
        update_cols = [c for c in cols if c != "index_id"]
        update_str = ", ".join([f"{c}=EXCLUDED.{c}" for c in update_cols]) + ", updated_at=NOW()"
        query = f"""
            INSERT INTO indices ({', '.join(cols)})
            VALUES %s
            ON CONFLICT (index_id) DO UPDATE SET {update_str}
        """
        return self._execute_batch(query, data)


    def upsert_securities(self, df: pd.DataFrame) -> int:
        """
        Вставляет информацию об активах в базу данных.

        Args:
            df: DataFrame с информацией об активах.

        Returns:
            Количество обработанных строк.
        """
        df = df.copy()
        df["ticker"] = df["SECID"]
        df["name"] = df["SECNAME"]
        df["short_name"] = df["SHORTNAME"]
        df["sector"] = df["SECTOR"].astype(str).replace({"nan": "", "None": ""})
        df["board"] = df["BOARDID"]
        df["market"] = df["MARKETCODE"]
        df["instrument_type"] = df["INSTRID"]
        df["security_type"] = df["SECTYPE"].astype(str).replace({"nan": "", "None": ""})
        df["currency"] = df["CURRENCYID"]
        df["lot_size"] = df["LOTSIZE"]
        df["min_step"] = df["MINSTEP"]
        df["decimals"] = df["DECIMALS"]
        df["isin"] = df["ISIN"]
        df["issue_size"] = df["ISSUESIZE"]
        df["listing_level"] = df["LISTLEVEL"]
        df["is_active"] = df["STATUS"] == "A"
        df["color"] = df["ticker"].apply(lambda x: self._color_from_hash(x))
        cols = ["ticker", "name", "short_name", "sector", "color", "board", "market",
                "instrument_type", "security_type", "currency", "lot_size", "min_step", "decimals",
                "isin", "issue_size", "listing_level", "is_active"]
        data = self._clean_df(df, cols)
        update_cols = [c for c in cols if c != "ticker"]
        update_str = ", ".join([f"{c}=EXCLUDED.{c}" for c in update_cols]) + ", updated_at=NOW()"
        query = f"""
            INSERT INTO securities ({', '.join(cols)})
            VALUES %s
            ON CONFLICT (ticker) DO UPDATE SET {update_str}
        """
        return self._execute_batch(query, data)


    def upsert_index_data(self, df: pd.DataFrame) -> int:
        """
        Вставляет данные по индексу в базу данных.

        Args:
            df: DataFrame с данными по индексу.
        
        Returns:
            Количество обработанных строк.
        """
        df['index_id'] = df['ticker']
        cols = ["date", "index_id", "open", "high", "low", "close", "volume", "value"]
        data = self._clean_df(df, cols)
        query = """
            INSERT INTO index_data (date, index_id, open, high, low, close, volume, value)
            VALUES %s
            ON CONFLICT (date, index_id) DO NOTHING
        """
        return self._execute_batch(query, data)
    

    def upsert_index_composition(self, df: pd.DataFrame) -> int:
        """
        Вставляет состав индекса в базу данных.

        Args:
            df: DataFrame с составом индекса.

        Returns:
            Количество обработанных строк.
        """
        df = df.copy()
        df["index_id"] = df["indexid"]
        df["date"] = pd.to_datetime(df["tradedate"]).dt.date if "tradedate" in df.columns else pd.Timestamp.today().date()
        df["ticker"] = df["ticker"]
        df["weight"] = df["weight"].astype(float) / 100.0
        
        cols = ["index_id", "date", "ticker", "weight"]
        data = self._clean_df(df, cols)

        query = """
            INSERT INTO index_composition (index_id, date, ticker, weight)
            VALUES %s
            ON CONFLICT (index_id, date, ticker) DO UPDATE SET weight = EXCLUDED.weight
        """
        return self._execute_batch(query, data)


    def upsert_market_data(self, df: pd.DataFrame) -> int:
        """
        Вставляет рыночные данные в базу данных.

        Args:
            df: DataFrame с рыночными данными.

        Returns:
            Количество обработанных строк.
        """
        cols = ["date", "ticker", "open", "high", "low", "close", "volume", "value"]
        data = self._clean_df(df, cols)
        query = """
            INSERT INTO market_data (date, ticker, open, high, low, close, volume, value)
            VALUES %s
            ON CONFLICT (date, ticker) DO NOTHING
        """
        return self._execute_batch(query, data)


    def upsert_risk_profiles(self, df: pd.DataFrame) -> int:
        """
        Вставляет или обновляет профили риска в базе данных.

        Args:
            df: DataFrame с информацией о профилях риска.

        Returns:
            Количество обработанных строк.
        """
        cols = ['display_name', 'description', 'is_active', 'rebalance_policy_planned', 'rebalance_policy_benchmark', 'rebalance_policy_weights', 'icon', 'model_name', 'eta']
        data = self._clean_df(df, cols)
        update_cols = [c for c in cols if c != "display_name"]
        update_str = ", ".join([f"{c}=EXCLUDED.{c}" for c in update_cols])
        query = f"""
            INSERT INTO risk_profiles ({', '.join(cols)})
            VALUES %s
            ON CONFLICT (profile_id) DO UPDATE SET {update_str}
        """
        return self._execute_batch(query, data)


    def _get_last_db_date(self, table: str, id_col: str = None, id_val: str = None, default_date: date = None) -> date:
        """
        Получает последнюю дату данных в указанной таблице БД.

        Args:
            table: Таблица для проверки.
            id_col: Столбец с идентификатором для фильтрации (опционально).
            id_val: Значение идентификатора для фильтрации (опционально).
            default_date: Дата по умолчанию, если таблица пуста (по умолчанию 2000-01-01).

        Returns:
            Последняя дата данных в таблице или дата по умолчанию, если таблица пуста.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                query = f"SELECT MAX(date) FROM {table}"
                params = ()
                if id_col and id_val:
                    query += f" WHERE {id_col} = %s"
                    params = (id_val,)
                cur.execute(query, params)
                res = cur.fetchone()
                if res and res[0]:
                    return res[0] if isinstance(res[0], date) else res[0].date()
        return default_date if default_date else date(2000, 1, 1)


    def get_risk_profile(self, profile_id: str) -> Optional[dict]:
        """
        Получает профиль риска по id.

        Args:
            profile_id: id профиля риска.

        Returns:
            Словарь с информацией о профиле риска.
        """
        import psycopg2.extras
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM risk_profiles WHERE profile_id = %s", (profile_id,))
                return cur.fetchone()
            

    def get_active_risk_profiles(self) -> Optional[pd.DataFrame]:
        """
        Получает все активные профили риска.

        Returns:
            DataFrame с информацией об активных профилях риска.
        """
        query = """
            SELECT *
            FROM risk_profiles
            WHERE is_active = true
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()            
        return pd.DataFrame(rows)
    

    def get_session(self, session_id: str) -> Optional[dict]:
        """
        Получает информацию о сессии по id.

        Args:
            session_id: id сессии.
        
        Returns:
            Словарь с информацией о сессии.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id.strip(),))
                row = cur.fetchone()
        return row
    

    def create_session(self, profile_id: str, balance: float, rebalance_policy_planned: str, rebalance_policy_benchmark: str, rebalance_policy_weights: str, created_at: date = None) -> str:
        """
        Создает новую сессию.

        Args:
            profile_id: id профиля риска.
            balance: Начальный баланс.
            rebalance_policy_planned: Частота плановой ребалансировки.
            rebalance_policy_benchmark: Допустимое отклонение бенчмарка.
            rebalance_policy_weights: Допустимое отклонение весов.
            created_at: Дата создания сессии (по умолчанию - сегодня).

        Returns:
            id новой сессии.
        """
        session_id = str(uuid.uuid4())
        if created_at is None:
            created_at = date.today()
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sessions (session_id, profile_id, current_balance, rebalance_policy_planned, rebalance_policy_benchmark, rebalance_policy_weights, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING session_id
                """, (session_id, profile_id, balance, rebalance_policy_planned, rebalance_policy_benchmark, rebalance_policy_weights, created_at))
                conn.commit()
                return cur.fetchone()[0]
           

    def update_session_params(self, session_id: str, balance: float = None, rebalance_policy_planned: int = None, rebalance_policy_benchmark: float = None, rebalance_policy_weights: float = None) -> None:
        """
        Обновляет параметры сессии.

        Args:
            session_id: id сессии.
            balance: Новый баланс.
            rebalance_policy_planned: Новая частота плановой ребалансировки.
            rebalance_policy_benchmark: Новое допустимое отклонение бенчмарка.
            rebalance_policy_weights: Новое допустимое отклонение весов.
        """
        updates = []
        params = []
        if balance is not None:
            updates.append("current_balance = %s")
            params.append(balance)
        if rebalance_policy_planned is not None:
            updates.append("rebalance_policy_planned = %s")
            params.append(rebalance_policy_planned)
        if rebalance_policy_benchmark is not None:
            updates.append("rebalance_policy_benchmark = %s")
            params.append(rebalance_policy_benchmark)
        if rebalance_policy_weights is not None:
            updates.append("rebalance_policy_weights = %s")
            params.append(rebalance_policy_weights)
        if updates:
            params.append(session_id)
            query = f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = %s"
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    conn.commit()


    def deactivate_session(self, session_id: str) -> None:
        """
        Деактивирует сессию.

        Args:
            session_id: id сессии.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET is_active = false WHERE session_id = %s",
                    (session_id,)
                )
                cur.execute(
                    "UPDATE portfolios SET status = 'closed', closed_at = NOW() WHERE session_id = %s AND status = 'active'",
                    (session_id,)
                )
                conn.commit()
            

    def get_portfolio(self, portfolio_id: str) -> Optional[dict]:
        """
        Получает информацию о портфеле по id.

        Args:
            portfolio_id: id портфеля.

        Returns:
            Словарь с информацией о портфеле.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM portfolios WHERE portfolio_id = %s",
                    (portfolio_id,)
                )
                return cur.fetchone()
            

    def get_active_portfolio(self, session_id: str) -> Optional[dict]:
        """
        Получает информацию об активном портфеле для сессии.

        Args:
            session_id: id сессии.

        Returns:
            Словарь с информацией об активном портфеле.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM portfolios WHERE session_id = %s AND status = 'active' LIMIT 1",
                    (session_id,)
                )
                return cur.fetchone()
            

    def create_portfolio(self, session_id: str, initial_balance: float, horizon_days: int, created_at: date = None, commission: float = 0.0) -> str:
        """
        Создает новый портфель для сессии.

        Args:
            session_id: id сессии.
            initial_balance: Начальный баланс портфеля.
            horizon_days: Горизонт инвестирования в торговых днях.
            created_at: Дата создания портфеля (по умолчанию - сегодня).
            commission: Комиссия при открытии портфеля.
        
        Returns:
            id нового портфеля.
        """
        portfolio_id = str(uuid.uuid4())
        if created_at is None:
            created_at = date.today()
        net_balance = initial_balance - commission
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute( """
                    INSERT INTO portfolios (portfolio_id, session_id, initial_balance, current_balance, horizon_days, created_at, status)
                    VALUES (%s, %s, %s, %s, %s, %s, 'active')
                    RETURNING portfolio_id
                """, (portfolio_id, session_id, net_balance, net_balance, horizon_days, created_at))
                conn.commit()
                return cur.fetchone()[0]
            

    def close_portfolio(self, portfolio_id: str, closed_at: date) -> None:
        """
        Закрывает портфель.

        Args:
            portfolio_id: id портфеля.
            closed_at: Дата закрытия портфеля.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE portfolios SET status = 'closed', closed_at = %s WHERE portfolio_id = %s",
                    (closed_at, portfolio_id)
                )
                conn.commit()
    

    def get_closed_portfolios(self, session_id: str) -> list[dict]:
        """
        Получает список закрытых портфелей для сессии.

        Args:
            session_id: id сессии.

        Returns:
            Список словарей с информацией о закрытых портфелях.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM portfolios WHERE session_id = %s AND status = 'closed' ORDER BY closed_at DESC",
                    (session_id,)
                )
                return cur.fetchall()
            

    def get_pending_rebalance(self, portfolio_id: str) -> Optional[dict]:
        """
        Получает информацию о ребалансировке в статусе "pending" для портфеля.

        Args:
            portfolio_id: id портфеля.

        Returns:
            Словарь с информацией о ребалансировке, включая дискретные позиции        
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT r.id, r.portfolio_id, r.rebalance_date, r.status, r.cash, r.reason, r.commission "
                    "FROM portfolio_rebalances r "
                    "WHERE r.portfolio_id = %s AND r.status = 'pending' "
                    "ORDER BY r.rebalance_date DESC LIMIT 1",
                    (portfolio_id,)
                )
                row = cur.fetchone()
                if not row:
                    return None
                cur.execute(
                    "SELECT ticker, lots FROM daily_portfolio_holdings dh "
                    "JOIN portfolio_daily_snapshots s ON dh.snapshot_id = s.id "
                    "WHERE s.portfolio_id = %s AND s.snapshot_date = %s",
                    (portfolio_id, row['rebalance_date'])
                )
                holdings_rows = cur.fetchall()
                row['discrete_holdings'] = {r['ticker']: r['lots'] for r in holdings_rows}
                return row
        
           
    def get_latest_portfolio_state(self, portfolio_id: str) -> Optional[dict]:
        """"
        Получает последний ежедневный снимок состояния портфеля.
        
        Args:
            portfolio_id: id портфеля.

        Returns:
            Словарь с информацией о состоянии портфеля, включая наличные, стоимость, значение бенчмарка z и дискретные позиции.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, snapshot_date, cash, portfolio_value, z_benchmark "
                    "FROM portfolio_daily_snapshots WHERE portfolio_id = %s "
                    "ORDER BY snapshot_date DESC LIMIT 1",
                    (portfolio_id,)
                )
                snap = cur.fetchone()
                if not snap: return None
                cur.execute(
                    "SELECT ticker, lots FROM daily_portfolio_holdings WHERE snapshot_id = %s",
                    (snap['id'],)
                )
                snap['holdings'] = {r['ticker']: r['lots'] for r in cur.fetchall()}
                return snap
            

    def get_portfolio_snapshot(self, portfolio_id: str, snapshot_date: date) -> Optional[dict]:
        """
        Получает ежедневный снимок состояния портфеля на заданную дату.

        Args:
            portfolio_id: id портфеля.
            snapshot_date: Дата снимка.

        Returns:
            Словарь с информацией о состоянии портфеля, включая наличные, стоимость, значение бенчмарка z и дискретные позиции.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, cash, portfolio_value, z_benchmark
                    FROM portfolio_daily_snapshots
                    WHERE portfolio_id = %s AND snapshot_date = %s
                """, (portfolio_id, snapshot_date))
                snap = cur.fetchone()
                if not snap:
                    return None
                cur.execute("SELECT ticker, lots FROM daily_portfolio_holdings WHERE snapshot_id = %s", (snap['id'],))
                snap['holdings'] = {r['ticker']: r['lots'] for r in cur.fetchall()}
                return snap
            

    def get_portfolio_history(self, portfolio_id: str, start_date: date, end_date: date) -> list[dict]:
        """
        Получает историю состояния портфеля за указанный период.

        Args:
            portfolio_id: id портфеля.
            start_date: Начальная дата.
            end_date: Конечная дата.

        Returns:
            Список словарей с информацией о состоянии портфеля.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT snapshot_date, portfolio_value, cash, z_benchmark FROM portfolio_daily_snapshots WHERE portfolio_id = %s AND snapshot_date BETWEEN %s AND %s ORDER BY snapshot_date", (portfolio_id, start_date, end_date))
                return cur.fetchall()
            
    
    def save_portfolio_snapshot(self, portfolio_id: str, snapshot_date: date, cash: float, portfolio_value: float, z_benchmark: float, holdings: dict[str, int]) -> None:
        """
        Сохраняет ежедневный снимок портфеля, включая наличные, стоимость портфеля, значение бенчмарка z и дискретные позиции.

        Args:
            portfolio_id: id портфеля.
            snapshot_date: Дата снимка.
            cash: Количество наличных.
            portfolio_value: Общая стоимость портфеля.
            z_benchmark: Значение бенчмарка z.
            holdings: Словарь с дискретными позициями.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO portfolio_daily_snapshots (portfolio_id, snapshot_date, cash, portfolio_value, z_benchmark)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (portfolio_id, snapshot_date) DO UPDATE SET
                        cash = EXCLUDED.cash, 
                        portfolio_value = EXCLUDED.portfolio_value, 
                        z_benchmark = EXCLUDED.z_benchmark
                    RETURNING id
                """, (portfolio_id, snapshot_date, cash, portfolio_value, z_benchmark))
                snap_id = cur.fetchone()['id']
                cur.execute("DELETE FROM daily_portfolio_holdings WHERE snapshot_id = %s", (snap_id,))
                if holdings:
                    psycopg2.extras.execute_values(
                        cur,
                        "INSERT INTO daily_portfolio_holdings (snapshot_id, ticker, lots) VALUES %s",
                        [(snap_id, t, l) for t, l in holdings.items()]
                    )
            conn.commit()
            

    def get_rebalance_history_detailed(self, portfolio_id: str) -> list[dict]:
        """
        Получает историю ребалансировок для портфеля.

        Args:
            portfolio_id: id портфеля.

        Returns:
            Список словарей с информацией о каждой ребалансировке, включая теоретические веса и дискретные позиции.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT r.id, r.rebalance_date, r.status, r.cash, r.commission, s.portfolio_value, "
                    "dh.ticker, dh.lots "
                    "FROM portfolio_rebalances r "
                    "LEFT JOIN portfolio_daily_snapshots s ON r.portfolio_id = s.portfolio_id AND r.rebalance_date = s.snapshot_date "
                    "LEFT JOIN daily_portfolio_holdings dh ON s.id = dh.snapshot_id "
                    "WHERE r.portfolio_id = %s "
                    "ORDER BY r.rebalance_date DESC",
                    (portfolio_id,)
                )
                rows = cur.fetchall()

            rebalances = {}
            for r in rows:
                rid = r['id']
                if rid not in rebalances:
                    rebalances[rid] = {
                        'id': rid, 'rebalance_date': r['rebalance_date'], 'status': r['status'],
                        'cash': float(r['cash']) if r['cash'] is not None else 0.0,
                        'commission': float(r['commission']) if r.get('commission') is not None else 0.0,
                        'portfolio_value': float(r['portfolio_value']) if r['portfolio_value'] is not None else None,
                        'theoretical_weights': {}, 'discrete_holdings': {}
                    }
                if r.get('ticker'):
                    rebalances[rid]['discrete_holdings'][r['ticker']] = int(r['lots'])

            if rebalances:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT rebalance_id, ticker, weight FROM rebalance_theoretical_weights WHERE rebalance_id IN %s",
                        (tuple(rebalances.keys()),)
                    )
                    for row in cur.fetchall():
                        if row['rebalance_id'] in rebalances:
                            rebalances[row['rebalance_id']]['theoretical_weights'][row['ticker']] = float(row['weight'])

            return list(rebalances.values())
            

    def save_rebalance_record(self, portfolio_id: str, date: str, theoretical_weights: dict, status: str = 'pending', cash: float = 0.0, reason: str = None, commission: float = 0.0, discrete_holdings: dict = None) -> None:
        """
        Сохраняет запись о ребалансировке портфеля.

        Args:
            portfolio_id: id портфеля.
            date: Дата ребалансировки.
            theoretical_weights: Словарь с теоретическими весами после ребалансировки.
            status: Статус ребалансировки (по умолчанию "pending").
            cash: Количество наличных после ребалансировки.
            reason: Причина ребалансировки.
            commission: Комиссия за ребалансировку.
            discrete_holdings: Словарь с дискретными позициями после ребалансировки (только для статуса "executed").
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO portfolio_rebalances (portfolio_id, rebalance_date, status, cash, reason, commission) "
                    "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id ",
                    (portfolio_id, date, status, cash, reason, commission)
                )
                rebalance_id = cur.fetchone()[0]
                if theoretical_weights:
                    psycopg2.extras.execute_values(
                        cur,
                        "INSERT INTO rebalance_theoretical_weights (rebalance_id, ticker, weight) VALUES %s ",
                        [(rebalance_id, t, w) for t, w in theoretical_weights.items()]
                    )
                if discrete_holdings and status == 'executed':
                    cur.execute("""
                        INSERT INTO portfolio_daily_snapshots (portfolio_id, snapshot_date, cash, portfolio_value, z_benchmark)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (portfolio_id, snapshot_date) DO UPDATE SET
                            cash = EXCLUDED.cash, portfolio_value = EXCLUDED.portfolio_value, z_benchmark = EXCLUDED.z_benchmark
                        RETURNING id
                    """, (portfolio_id, date, cash, cash + sum(l * 1 for l in discrete_holdings.values()), 1.0))
                    snap_id = cur.fetchone()[0]
                    psycopg2.extras.execute_values(
                        cur,
                        "INSERT INTO daily_portfolio_holdings (snapshot_id, ticker, lots) VALUES %s",
                        [(snap_id, t, l) for t, l in discrete_holdings.items()]
                    )
            conn.commit()


    def update_rebalance_status(self, rebalance_id: str, status: str) -> None:
        """
        Обновляет статус ребалансировки.

        Args:
            rebalance_id: id ребалансировки.
            status: Новый статус ребалансировки.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE portfolio_rebalances SET status = %s WHERE id = %s",
                    (status, rebalance_id)
                )
                conn.commit()
    
            
    def update_z(self, portfolio_id: str, current_z: float) -> None:
        """
        Обновляет значение бенчмарка z для портфеля.

        Args:
            portfolio_id: id портфеля.
            current_z: Текущее значение бенчмарка z.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE portfolios
                    SET last_z = %s
                    WHERE portfolio_id = %s
                """, (current_z, portfolio_id))
                conn.commit()


    def update_current_balance(self, portfolio_id: str, session_id: str, new_balance: float) -> None:
        """
        Обновляет текущий баланс портфеля и сессии.

        Args:
            portfolio_id: id портфеля.
            session_id: id сессии.
            new_balance: Новый баланс.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE portfolios SET current_balance = %s WHERE portfolio_id = %s", (new_balance, portfolio_id))
                cur.execute("UPDATE sessions SET current_balance = %s WHERE session_id = %s", (new_balance, session_id))
            conn.commit()
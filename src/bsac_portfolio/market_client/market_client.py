"""Клиент для получения рыночных данных MOEX ISS."""
import re
from datetime import timedelta
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

SECTOR_MAPPING = {
    "НЕФТЕГАЗ": "Нефтегазовая промышленность",
    "БАНКИ": "Банковский сектор",
    "ФИНАНСЫ": "Финансовый сектор",
    "МЕТАЛЛУРГИЯ черн.": "Чёрная металлургия",
    "МЕТАЛЛУРГИЯ цвет.": "Цветная металлургия",
    "МЕТАЛЛУРГИЯ разное": "Металлургия (разное)",
    "ДРАГ.МЕТАЛЛЫ": "Драгоценные металлы",
    "ГОРНОДОБЫВАЮЩИЕ": "Горнодобывающая промышленность",
    "ХИМИЯ удобрения": "Химическая промышленность (удобрения)",
    "ХИМИЯ разное": "Химическая промышленность (разное)",
    "Э/ГЕНЕРАЦИЯ": "Электроэнергетика (генерация)",
    "ЭЛЕКТРОСЕТИ": "Электросетевой комплекс",
    "ЭНЕРГОСБЫТ": "Энергосбыт",
    "РИТЕЙЛ": "Розничная торговля",
    "ПОТРЕБ": "Потребительский сектор",
    "Агропром и Пищепром": "Агропромышленный комплекс и пищевая промышленность",
    "Промышленность разное": "Промышленность (разное)",
    "ТЕЛЕКОМ": "Телекоммуникации",
    "ИНТЕРНЕТ": "Интернет-компании",
    "HIGH TECH": "Высокие технологии",
    "Производство Софта": "Разработка программного обеспечения",
    "Фармацевтика": "Фармацевтическая промышленность",
    "МЕДИА": "Медиа",
    "ТРАНСПОРТ": "Транспортный сектор",
    "СТРОИТЕЛИ": "Строительная отрасль",
    "МАШИНОСТРОЕНИЕ": "Машиностроение",
    "ТРЕТИЙ ЭШЕЛОН": "Третий эшелон",
    "НЕПУБЛИЧНЫЕ": "Непубличные компании",
    "ДРУГОЕ": "Прочие сектора"
}
MOEX_URL = "https://iss.moex.com/iss"
SMARTLAB_URL = "https://smart-lab.ru"


class MarketDataClient:
    """
    Клиент для загрузки данных с MOEX ISS.
    """
    def __init__(self, timeout: int = 30, max_retries: int = 3, retry_delay: int = 2) -> None:
        """
        Инициализация клиента MOEX ISS.
        
        Args:
            timeout: Таймаут запроса в секундах.
            max_retries: Максимальное количество попыток повторного запроса.
            retry_delay: Базовая задержка между попытками в секундах.
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})


    def _request(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """
        Выполнение HTTP GET запроса с повторными попытками.
        
        Args:
            url: URL для запроса.
            params: Параметры запроса.
            
        Returns:
            JSON ответ сервера.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f'Запрос не удался (попытка {attempt+1}/{self.max_retries}): {e}')
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise


    @staticmethod
    def split_date_range(start: str, end: str, max_days: int = 365) -> list:
        """
        Разбивает диапазон дат на интервалы.
        
        Args:
            start: Начальная дата.
            end: Конечная дата.
            max_days: Максимальное количество дней в интервале.
            
        Returns:
            Список кортежей с датами начала и конца интервалов.
        """
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        intervals = []
        current = start_dt
        while current <= end_dt:
            chunk_end = min(current + timedelta(days=max_days-1), end_dt)
            intervals.append((current.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
            current = chunk_end + timedelta(days=1)
        return intervals
    

    def fetch_sector_name(self, ticker: str, retry: bool = False) -> Optional[str]:
        """
        Загружает названия сектора для тикера.
        
        Args:
            ticker: Тикер компании.
            retry: Флаг повторной попытки для ticker[:-1].
            
        Returns:
            Название сектора или None.
        """
        url = f"{SMARTLAB_URL}/q/{ticker.upper()}/f/y/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            results = str(soup.find_all(string=re.compile("сектора"))[0])
            return SECTOR_MAPPING.get(results.lstrip("Aнализ сектора "), None)
        except Exception as e:
            print(f"Ошибка при получении сектора для {ticker}")
            if retry and ticker[-1] == "P":
                try:
                    print(f"Повторная попытка для {ticker[:-1]}")
                    return self.fetch_sector_name(ticker[:-1], retry=False)
                except:
                    pass
            else:
                return "NaN"
        return None


    def fetch_candles(self, ticker: str, start: str, end: str, interval: int = 24, engine: str = 'stock', market: str = 'shares') -> pd.DataFrame:
        """
        Загружает свечные данные для тикера.
        
        Args:
            ticker: Тикер инструмента.
            start: Начальная дата.
            end: Конечная дата.
            interval: Интервал свечей.
            engine: Тип рынка.
            market: Сегмент рынка.
            
        Returns:
            DataFrame со свечными данными.
        """
        intervals = self.split_date_range(start, end, max_days=365)
        all_data = []
        url = f"{MOEX_URL}/engines/{engine}/markets/{market}/securities/{ticker}/candles.json"
        for from_d, till_d in intervals:
            params = {
                'from': from_d,
                'till': till_d,
                'interval': interval,
                'iss.meta': 'off',
                'iss.json': 'compact',
                'limit': 100,
                'start': 0
            }
            while True:
                data = self._request(url, params)
                candles = data.get('candles', {})
                rows = candles.get('data', [])
                if not rows:
                    break
                
                df_chunk = pd.DataFrame(rows, columns=candles['columns'])
                all_data.append(df_chunk)
                
                if len(rows) < params['limit']:
                    break
                params['start'] += params['limit']
        if not all_data:
            print(f'Данные по свечам для {ticker} не найдены.')
            return pd.DataFrame()
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=['begin']).sort_values('begin').reset_index(drop=True)
        df = df.rename(columns={
            'open': 'open',
            'close': 'close',
            'high': 'high',
            'low': 'low',
            'volume': 'volume',
            'value': 'value',
            'begin': 'date'
        })
        df['date'] = pd.to_datetime(df['date'])
        df['ticker'] = ticker
        return df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'value']]


    def fetch_index_structure(self, index_id: str, date: str) -> pd.DataFrame:
        """
        Загружает структуру индекса на дату.
        
        Args:
            index_id: id индекса.
            date: Дата для получения структуры.
            
        Returns:
            DataFrame со структурой индекса.
        """
        url = f"{MOEX_URL}/statistics/engines/stock/markets/index/analytics/{index_id}.json"
        params = {
            'iss.meta': 'off',
            'iss.json': 'compact',
            'limit': 1000,
            'date': date
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        analytics = data.get('analytics', {})
        if not analytics.get('data'):
            return pd.DataFrame()
        df = pd.DataFrame(analytics['data'], columns=analytics['columns'])
        if 'tradedate' in df.columns:
            df['tradedate'] = pd.to_datetime(df['tradedate'])
        return df
    

    def fetch_indices_list(self) -> pd.DataFrame:
        """
        Загружает список индексов MOEX.
        
        Returns:
            DataFrame со списком индексов.
        """
        url = f"{MOEX_URL}/engines/stock/markets/index/securities.json"
        params = {'iss.meta': 'off', 'iss.json': 'compact', 'limit': 500}
        
        data = self._request(url, params)
        sec = data.get('securities', {})
        rows = sec.get('data', [])
        
        if not rows:
            return pd.DataFrame()
            
        df = pd.DataFrame(rows, columns=sec['columns'])
        return df.drop_duplicates(subset=['SECID']).sort_values('SECID').reset_index(drop=True)


    def fetch_ticker_list(self, engine: str = 'stock', market: str = 'shares', board: str = 'TQBR', fetch_sectors: bool = True) -> pd.DataFrame:
        """
        Загружает список тикеров.
        
        Args:
            engine: Тип рынка.
            market: Сегмент рынка.
            board: Режим торгов.
            fetch_sectors: Загружать ли информацию о секторах.
            
        Returns:
            DataFrame со списком тикеров.
        """
        if not board:
            raise ValueError("Параметр 'board' обязателен для этого запроса (например, 'TQBR')")
        url = f"{MOEX_URL}/engines/{engine}/markets/{market}/boards/{board}/securities.json"
        params = {
            'iss.meta': 'off',
            'iss.json': 'compact',
            'limit': 100,
            'start': 0
        }
        all_data = []
        first_row_id = None
        while True:
            data = self._request(url, params)
            cursor = data.get('securities.cursor') or data.get('cursor')
            if cursor and cursor.get('data'):
                cursor_row = cursor['data'][0]
                index, total, pagesize = cursor_row[0], cursor_row[1], cursor_row[2]
                securities = data.get('securities', {})
                rows = securities.get('data', [])
                if rows:
                    cols = securities.get('columns')
                    df_chunk = pd.DataFrame(rows, columns=cols)
                    all_data.append(df_chunk)
                if index + pagesize < total:
                    params['start'] = index + pagesize
                    continue
                else:
                    break
            securities = data.get('securities', {})
            rows = securities.get('data', [])
            if not rows:
                break
            cols = securities.get('columns')
            df_chunk = pd.DataFrame(rows, columns=cols)
            current_first_id = df_chunk.iloc[0]['SECID'] if 'SECID' in df_chunk.columns else None
            if first_row_id is not None and current_first_id == first_row_id:
                break
            if first_row_id is None:
                first_row_id = current_first_id
            all_data.append(df_chunk)
            if len(rows) < params['limit']:
                break
            params['start'] += params['limit']
        if not all_data:
            print(f"Не найдено данных для engine={engine}, market={market}, board={board}")
            return pd.DataFrame()
        all_data = pd.concat(all_data, ignore_index=True)
        all_data["SECTORS"] = None
        if fetch_sectors:
            sectors = []
            for ticker in all_data["SECID"].tolist():
                sector = self.fetch_sector_name(ticker)
                sectors.append(sector)
            all_data['SECTOR'] = sectors
        return all_data
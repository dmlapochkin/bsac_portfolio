"""Логика расчета метрик и ребалансировки портфеля."""
import numpy as np
import pandas as pd
from typing import Optional, Any


def calculate_asset_metrics(price_series: pd.Series, returns_series: pd.Series) -> Optional[dict[str, float]]:
    """
    Рассчитывает метрики для актива.

    Args:
        price_series: Серия цен актива.
        returns_series: Серия доходностей актива.

    Returns:
        Словарь с метриками.
    """
    if len(returns_series) < 2:
        return None
    cumulative = (1 + returns_series).cumprod()
    total_return = cumulative.iloc[-1] - 1
    days = len(returns_series)
    years = days / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annualized_vol = returns_series.std() * np.sqrt(252)
    sharpe = (annualized_return) / annualized_vol if annualized_vol > 0 else 0
    rolling_max = price_series.expanding().max()
    drawdown = (price_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    var_95 = np.percentile(returns_series.dropna(), 5)
    cvar_95 = returns_series[returns_series <= var_95].mean()
    return {
        'total_return': total_return, 'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol, 'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown, 'var_95': var_95, 'cvar_95': cvar_95,
    }


def calculate_portfolio_metrics(executed: list[dict[str, Any]], market_data: pd.DataFrame, initial_balance: float, lot_sizes: dict[str, int]) -> dict[str, float]:
    """
    Рассчитывает метрики для портфеля.

    Args:
        executed: Список словарей с данными о выполненных ребалансировках.
        market_data: DataFrame с рыночными данными.
        initial_balance: Начальный баланс портфеля.
        lot_sizes: Словарь с размером лота для каждого тикера.

    Returns:
        Словарь с метриками портфеля.
    """
    if not executed:
        return {'total_return': 0.0, 'volatility': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0, 'var_95': 0.0, 'cvar_95': 0.0}
    
    portfolio_values = []
    for h in executed:
        val = h.get('portfolio_value')
        if val is not None:
            portfolio_values.append(float(val))
        else:
            r_date = pd.to_datetime(h['rebalance_date']).date()
            holdings = h.get('discrete_holdings', {})
            cash = float(h.get('cash', 0.0)) if h.get('cash') is not None else 0.0
            prices = {row['ticker']: float(row['close']) for _, row in market_data[market_data['date'] == r_date].iterrows()}
            asset_value = sum(lots * prices.get(t, 0.0) * lot_sizes.get(t, 1) for t, lots in holdings.items())
            portfolio_values.append(cash + asset_value)
    if len(portfolio_values) < 2:
        total_ret = (portfolio_values[-1] / initial_balance - 1.0) if portfolio_values else 0.0
        return {'total_return': total_ret, 'volatility': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0, 'var_95': 0.0, 'cvar_95': 0.0}
    values_series = pd.Series(portfolio_values)
    returns = values_series.pct_change().dropna()
    total_return = values_series.iloc[-1] / initial_balance - 1.0
    n_days = len(returns)
    ann_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0
    volatility = returns.std() * np.sqrt(252)
    sharpe = ann_return / volatility if volatility > 0 else 0.0
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    var_95 = np.percentile(returns, 5)
    tail = returns[returns <= var_95]
    cvar_95 = tail.mean() if not tail.empty else var_95
    return {
        'total_return': total_return, 'volatility': volatility, 'sharpe': sharpe,
        'max_drawdown': max_dd, 'var_95': var_95, 'cvar_95': cvar_95
    }


def calculate_actual_weights(holdings: dict[str, int], market_data: dict[str, dict]) -> dict[str, float]:
    """
    Рассчитывает фактические веса активов в портфеле.

    Args:
        holdings: Словарь с количеством лотов по каждому тикеру.
        market_data: Словарь с рыночными данными по каждому тикеру.
    
    Returns:
        Словарь с фактическими весами активов в портфеле.
    """
    weights = {}
    total = 0.0
    for t, lots in holdings.items():
        if t in market_data:
            val = lots * market_data[t]['close'] * market_data[t]['lot_size']
            weights[t] = val
            total += val
    return {k: v / total for k, v in weights.items()} if total > 0 else weights


def calculate_current_z(last_z: float, daily_returns: list[float], eta: float, risk_free_daily: float = 0.0002) -> float:
    """
    Рассчитывает текущее значение z (нормированного бенчмарка) на основе последних доходностей.

    Args:
        last_z: Предыдущее значение z.
        daily_returns: Список ежедневных доходностей портфеля.
        eta: Параметр адаптивности профиля риска.
        risk_free_daily: Ежедневная безрисковая ставка.

    Returns:
        Текущее значение z.
    """
    z = last_z
    for r in daily_returns:
        z = eta * z * ((1.0 + risk_free_daily) / (1.0 + r)) + (1.0 - eta)
    return z


def check_rebalance_criteria(current_holdings: dict[str, int], last_theoretical_weights: dict[str, float], market_data: dict[str, dict], current_z: float, last_z: float, days_since_last: int, policy_weights: float, policy_benchmark: float, policy_planned: int) -> tuple[bool, list[str]]:
    """
    Проверяет критерии ребалансировки портфеля.

    Args:
        current_holdings: Словарь с текущими позициями по каждому тикеру.
        last_theoretical_weights: Словарь с последними теоретическими весами активов.
        market_data: Словарь с рыночными данными по каждому тикеру.
        current_z: Текущее значение z.
        last_z: Предыдущее значение z.
        days_since_last: Количество дней с последней ребалансировки.
        policy_weights: Допустимое отклонение весов.
        policy_benchmark: Допустимое отклонение бенчмарка.
        policy_planned: Частота плановой ребалансировки.

    Returns:
        Кортеж из двух элементов: (нужно ли ребалансировать, список триггеров).
    """
    actual_w = calculate_actual_weights(current_holdings, market_data)
    all_tickers = set(list(actual_w.keys()) + list(last_theoretical_weights.keys()))
    max_dev = max(abs(actual_w.get(t, 0.0) - last_theoretical_weights.get(t, 0.0)) for t in all_tickers) if all_tickers else 0.0
    triggers = []
    if max_dev > policy_weights:
        triggers.append("отклонение весов")
    if abs(current_z - last_z) > policy_benchmark:
        triggers.append("отклонение бенчмарка")
    if days_since_last >= policy_planned:
        triggers.append("плановая ребалансировка")
    return len(triggers) > 0, triggers


def calculate_forecast_metrics(cum_paths: np.ndarray, start_val: float, horizon_days: int, risk_free_rate: float = 0.00) -> dict[str, float]:
    """
    Рассчитывает прогнозные метрики портфеля.

    Args:
        cum_paths: Массив с накопленными значениями портфеля.
        start_val: Начальное значение портфеля.
        horizon_days: Количество дней горизонта прогноза.
        risk_free_rate: Безрисковая ставка.

    Returns:
        Словарь с прогнозными метриками.
    """
    if cum_paths is None or len(cum_paths) == 0:
        return {}
    
    final_values = cum_paths[:, -1]
    final_returns = (final_values - start_val) / start_val
    
    expected_return = np.mean(final_returns)
    expected_annual_return = (1 + expected_return) ** (252 / horizon_days) - 1 if horizon_days > 0 else 0.0
    
    volatility = np.std(final_returns) * np.sqrt(252 / horizon_days) if horizon_days > 0 else 0.0
    sharpe = (expected_annual_return - risk_free_rate) / volatility if volatility > 0 else 0.0
    
    var_95 = np.percentile(final_returns, 5)
    tail = final_returns[final_returns <= var_95]
    cvar_95 = np.mean(tail) if len(tail) > 0 else var_95
    
    downside_returns = final_returns[final_returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(252 / horizon_days) if len(downside_returns) > 0 else 0.0
    sortino = (expected_annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0
    
    return {
        'expected_return': expected_return,
        'expected_annual_return': expected_annual_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'sortino': sortino,
        'var_95': var_95,
        'cvar_95': cvar_95
    }


def discretize_portfolio(target_weights: dict[str, float], market_data: dict[str, dict], cash: float, commission_rate: float = 0.0, previous_holdings: dict[str, int] = None, min_weight: float = 0.005) -> tuple[dict[str, int], float, float]:
    """
    Дискретизирует портфель, округляя количество лотов до целых значений.

    Args:
        target_weights: Словарь с целевыми весами активов.
        market_data: Словарь с рыночными данными по каждому тикеру.
        cash: Свободные денежные средства.
        commission_rate: Комиссия.
        previous_holdings: Словарь с предыдущими позициями по каждому тикеру.
        min_weight: Минимальный вес для включения актива в портфель.

    Returns:
        Кортеж из трех элементов: (новые позиции по каждому тикеру, оставшиеся денежные средства, стоимость комиссий).
    """
    holdings = {}
    if previous_holdings is None:
        previous_holdings = {}

    for ticker, w in sorted(target_weights.items(), key=lambda x: x[1], reverse=True):
        if w < min_weight or ticker not in market_data:
            continue
        info = market_data[ticker]
        lot_cost = float(info['close']) * int(info['lot_size'])
        if lot_cost <= 0:
            continue
        
        target_lots = int((w * cash) // lot_cost)
        if target_lots > 0:
            holdings[ticker] = target_lots

    turnover = 0.0
    for t, lots in holdings.items():
        prev_lots = previous_holdings.get(t, 0)
        diff = abs(lots - prev_lots)
        if diff > 0 and t in market_data:
            turnover += diff * market_data[t]['close'] * market_data[t]['lot_size']
            
    commission_amount = turnover * commission_rate
    
    new_holdings_value = sum(
        l * market_data[t]['close'] * market_data[t]['lot_size'] 
        for t, l in holdings.items()
    )
    remaining = cash - new_holdings_value - commission_amount
    
    return holdings, remaining, commission_amount


def calculate_rebalance_actions(current_holdings: dict, previous_holdings: dict, market_data: dict = None) -> pd.DataFrame:
    """
    Рассчитывает действия по ребалансировке, сравнивая текущие позиции с предыдущими.

    Args:
        current_holdings: Словарь с текущими позициями по каждому тикеру.
        previous_holdings: Словарь с предыдущими позициями по каждому тикеру.
        market_data: Словарь с рыночными данными по каждому тикеру (для расчета стоимости).

    Returns:
        DataFrame с действиями по ребалансировке.
    """
    all_tickers = set(list(current_holdings.keys()) + list(previous_holdings.keys()))
    actions = []
    total_turnover = 0.0
    for ticker in all_tickers:
        curr_lots = current_holdings.get(ticker, 0)
        prev_lots = previous_holdings.get(ticker, 0)
        diff = curr_lots - prev_lots
        if diff != 0:
            action_cost = 0.0
            if market_data and ticker in market_data:
                info = market_data[ticker]
                lot_cost = float(info['close']) * int(info['lot_size'])
                action_cost = abs(diff) * lot_cost
                total_turnover += action_cost
                actions.append({"Тикер": ticker, "Действие": "Покупка" if diff > 0 else "Продажа", "Кол-во лотов": abs(diff), "Текущее кол-во": prev_lots, "Целевое кол-во": curr_lots, "Цена": lot_cost, "Стоимость": action_cost})
    df = pd.DataFrame(actions).sort_values(by=["Действие", "Тикер"]) if actions else pd.DataFrame()
    return df, total_turnover


def run_monte_carlo_forecast(holdings_dict: dict[str, int], market_data: dict[str, dict], historical_returns: pd.DataFrame, horizon_days: int = 60, n_simulations: int = 200) -> Optional[dict]:
    """
    Выполняет моделирование Монте-Карло для прогноза стоимости портфеля.

    Args:
        holdings_dict: Словарь с количеством лотов по каждому тикеру.
        market_data: Словарь с рыночными данными по каждому тикеру.
        historical_returns: DataFrame с историческими доходностями по каждому тикеру.
        horizon_days: Количество дней горизонта прогноза.
        n_simulations: Количество симуляций.

    Returns:
        Словарь с результатами моделирования: накопленные пути, 5-й, 50-й и 95-й перцентили, прогнозная стоимость.
    """
    valid_tickers = [t for t in holdings_dict if t in market_data and market_data[t]['close'] > 0]
    if not valid_tickers or historical_returns.empty:
        return None
        
    weights = {}
    total_val = 0
    for t in valid_tickers:
        cost = holdings_dict[t] * market_data[t]['close'] * market_data[t]['lot_size']
        total_val += cost
        weights[t] = cost
        
    if total_val == 0:
        return None
        
    weights = {t: v / total_val for t, v in weights.items()}
    valid_cols = [t for t in historical_returns.columns if t in valid_tickers]
    ret_pivot = historical_returns[valid_cols].dropna()
    
    if ret_pivot.empty or len(ret_pivot) < horizon_days:
        return None
        
    mu = ret_pivot.mean().values
    sigma = ret_pivot.std().values
    cov = ret_pivot.cov().values
    
    try:
        L = np.linalg.cholesky(cov + np.eye(len(valid_cols)) * 1e-6)
    except np.linalg.LinAlgError:
        return None
        
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(n_simulations, horizon_days, len(valid_cols)))
    correlated_shocks = Z @ L.T
    
    paths = np.zeros((n_simulations, horizon_days + 1, len(valid_cols)))
    paths[:, 0, :] = 1.0
    
    for t in range(1, horizon_days + 1):
        paths[:, t, :] = paths[:, t-1, :] * np.exp(mu + correlated_shocks[:, t-1, :])
        
    weight_vector = np.array([weights.get(t, 0) for t in ret_pivot.columns])
    port_paths = np.sum(paths * weight_vector[np.newaxis, np.newaxis, :], axis=2)
    cum_paths = total_val * port_paths
    
    return {
        'cum_paths': cum_paths,
        'p5': np.percentile(cum_paths, 5, axis=0),
        'p50': np.percentile(cum_paths, 50, axis=0),
        'p95': np.percentile(cum_paths, 95, axis=0),
        'start_val': total_val
    }
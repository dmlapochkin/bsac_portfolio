"""Модуль управления датой и календарём."""

from datetime import date, timedelta

import exchange_calendars as xcals
import pandas as pd
import streamlit as st

def _get_xmos_calendar() -> xcals.ExchangeCalendar:
    """
    Возвращает календарь Московской биржи с кэшированием.
    
    Returns:
        Экземпляр календаря Московской биржи
    """
    return xcals.get_calendar("XMOS")


def is_trading_day(target_date: date) -> bool:
    """
    Проверяет, является ли указанная дата торговым днём.

    Args:
        target_date: Проверяемая дата

    Returns:
        True, если дата является торговым днём, иначе False
    """
    cal = _get_xmos_calendar()
    return cal.is_session(pd.Timestamp(target_date))


def get_next_trading_day(target_date: date) -> date:
    """
    Возвращает следующий торговый день после указанной даты.

    Args:
        target_date: Проверяемая дата

    Returns:
        Следующий торговый день
    """
    cal = _get_xmos_calendar()
    ts = pd.Timestamp(target_date)
    if cal.is_session(ts):
        return target_date
    return cal.next_session(ts).date()


def get_trading_days_range(start_date: date, n_days: int) -> pd.DatetimeIndex:
    """
    Генерирует диапазон торговых дней начиная с указанной даты.

    Args:
        start_date: Начальная дата
        n_days: Количество торговых дней

    Returns:
        Диапазон торговых дней
    """
    cal = _get_xmos_calendar()
    start_ts = pd.Timestamp(start_date)
    end_guess = start_ts + pd.Timedelta(days=int(n_days * 1.6))
    sessions = cal.sessions_in_range(start_ts, end_guess)
    sessions = sessions[sessions >= start_ts]
    return sessions[:n_days]


def _get_next_trading_session(target_date: date) -> date:
    """
    Возвращает следующую торговую сессию после указанной даты.

    Args:
        target_date: Проверяемая дата

    Returns:
        Следующая торговая сессия
    """
    cal = _get_xmos_calendar()
    try:
        return cal.next_session(pd.Timestamp(target_date)).date()
    except Exception:
        return target_date


def _get_prev_trading_session(target_date: date) -> date:
    """
    Возвращает предыдущую торговую сессию перед указанной датой.

    Args:
        target_date: Проверяемая дата

    Returns:
        Предыдущая торговая сессия
    """
    cal = _get_xmos_calendar()
    try:
        return cal.previous_session(pd.Timestamp(target_date)).date()
    except Exception:
        return target_date


def get_active_date() -> date:
    """
    Возвращает активную дату приложения с учётом демо-режима.

    Returns:
        Активная дата веб-приложения
    """
    if st.session_state.get("demo_date_override"):
        return st.session_state.demo_date_override
    from .config import APP_MODE, APP_DEMO_DATE
    if APP_MODE == "demo" and APP_DEMO_DATE:
        try:
            if isinstance(APP_DEMO_DATE, str):
                return date.fromisoformat(APP_DEMO_DATE)
            return APP_DEMO_DATE
        except (ValueError, TypeError):
            pass
    return date.today()


def _change_date(days: int) -> None:
    """
    Изменяет текущую дату на указанное количество дней.
    
    Args:
        days: Количество дней для изменения
    """
    active_p = None
    session_id = st.session_state.get('session_id')
    if session_id:
        from .data_cache import db
        active_p = db.get_active_portfolio(session_id)
    if active_p:
        advance_simulation_day(active_p['portfolio_id'], days)
    else:
        st.session_state.demo_date_override = get_active_date() + timedelta(days=days)
        st.cache_data.clear()


def _reset_date() -> None:
    """Сбрасывает переопределение даты демо-режима."""
    st.session_state.pop("demo_date_override", None)
    st.cache_data.clear()


def advance_simulation_day(portfolio_id: str, direction: int) -> None:
    """
    Перемещает симуляцию на один торговый день вперёд или назад.
    
    Args:
        portfolio_id: id портфеля
        direction: Направление изменения
    """
    from .config import RISKFREE_RATE
    
    current_date = get_active_date()
    target_date = _get_next_trading_session(current_date) if direction > 0 else _get_prev_trading_session(current_date)
    if target_date == current_date:
        return

    from .data_cache import db
    base_snap = db.get_latest_portfolio_state(portfolio_id)
    if not base_snap:
        st.session_state.demo_date_override = target_date
        st.cache_data.clear()
        st.rerun()
        return

    holdings = base_snap['holdings']
    cash = float(base_snap['cash'])
    current_z = float(base_snap['z_benchmark'])
    profile = db.get_risk_profile(st.session_state.risk_profile_id)
    eta = float(profile.get('eta', 1.0))
    rf_daily = RISKFREE_RATE
    tickers = list(holdings.keys())
    market_data = db.get_latest_market_data(tickers, target_date=target_date)
    if not market_data:
        st.session_state.demo_date_override = target_date
        st.cache_data.clear()
        st.rerun()
        return

    invested_val = sum(l * market_data[t]['close'] * market_data[t]['lot_size'] for t, l in holdings.items() if t in market_data)
    total_value = cash + invested_val
    prev_snap = db.get_portfolio_snapshot(portfolio_id, base_snap['snapshot_date'])
    if prev_snap and prev_snap['portfolio_value'] > 0:
        daily_return = (total_value - float(prev_snap['portfolio_value'])) / float(prev_snap['portfolio_value'])
        new_z = eta * current_z * ((1.0 + rf_daily) / (1.0 + daily_return)) + (1.0 - eta)
    else:
        new_z = current_z

    db.save_portfolio_snapshot(portfolio_id, target_date, cash, total_value, new_z, holdings)
    db.update_current_balance(portfolio_id, st.session_state.session_id, total_value)
    st.session_state.demo_date_override = target_date
    st.cache_data.clear()
    st.rerun()


def _fill_portfolio_history(portfolio_id: str, target_date: date) -> None:
    """
    Заполняет историю портфеля данными за пропущенные торговые дни.
    
    Args:
        portfolio_id: id портфеля
        target_date: Целевая дата для заполнения истории
    """
    from .config import RISKFREE_RATE
    from .data_cache import db
    last_snap = db.get_latest_portfolio_state(portfolio_id)
    if not last_snap or last_snap['snapshot_date'] >= target_date:
        return

    cal = _get_xmos_calendar()
    sessions = cal.sessions_in_range(pd.Timestamp(last_snap['snapshot_date']), pd.Timestamp(target_date))
    days_to_fill = [d.date() for d in sessions if d.date() > last_snap['snapshot_date']]

    holdings = last_snap['holdings']
    cash = float(last_snap['cash'])
    current_z = float(last_snap['z_benchmark'])
    prev_val = float(last_snap['portfolio_value'])
    tickers = list(holdings.keys())
    profile = db.get_risk_profile(st.session_state.risk_profile_id)
    eta = float(profile.get('eta', 1.0)) if profile else 1.0
    rf_daily = RISKFREE_RATE

    for day in days_to_fill:
        md = db.get_latest_market_data(tickers, target_date=day)
        if not md:
            continue
        inv = sum(l * md[t]['close'] * md[t]['lot_size'] for t, l in holdings.items() if t in md)
        total = cash + inv
        ret = (total - prev_val) / prev_val if prev_val > 0 else 0.0
        z = eta * current_z * ((1.0 + rf_daily) / (1.0 + ret)) + (1.0 - eta)
        db.save_portfolio_snapshot(portfolio_id, day, cash, total, z, holdings)
        current_z, prev_val = z, total

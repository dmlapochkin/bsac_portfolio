"""Модуль управления сессиями и портфелями."""

from datetime import timedelta

import pandas as pd
import streamlit as st
from streamlit_extras.card_selector import card_selector

from bsac_portfolio.database.db_manager import DatabaseManager
from bsac_portfolio.portfolio.portfolio_logic import check_rebalance_criteria, calculate_current_z
from .config import APP_MODE


def _check_portfolio_triggers(portfolio_id: str) -> None:
    """
    Проверяет триггеры ребалансировки для активного портфеля.
    
    Args:
        portfolio_id: id портфеля
    """
    from .data_cache import db
    from .date_utils import get_active_date, get_trading_days_range
    portfolio = db.get_portfolio(portfolio_id)
    if not portfolio or portfolio.get('status') == 'closed':
        return

    current_date = get_active_date()
    history = db.get_rebalance_history_detailed(portfolio_id)
    executed = [h for h in history if h.get('status') == 'executed']
    if not executed:
        return

    last_rec = executed[0]
    profile = db.get_risk_profile(st.session_state.risk_profile_id)
    if not profile:
        return

    tickers = list(last_rec.get('discrete_holdings', {}).keys())
    market_data = db.get_latest_market_data(tickers, target_date=current_date)

    current_val = float(last_rec.get('cash', 0.0))
    for t, lots in last_rec.get('discrete_holdings', {}).items():
        info = market_data.get(t)
        if info:
            current_val += lots * info['close'] * info['lot_size']

    hist_returns = db.get_market_data(tickers, pd.to_datetime(last_rec['rebalance_date']).date().strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
    daily_rets = []
    if not hist_returns.empty and 'close' in hist_returns.columns:
        hist_returns['close'] = pd.to_numeric(hist_returns['close'], errors='coerce')
        ret_pivot = hist_returns.pivot(index='date', columns='ticker', values='close').pct_change().dropna()
        weights = last_rec.get('theoretical_weights', {})
        daily_rets = ret_pivot.astype(float).dot(pd.Series({k: float(v) for k, v in weights.items()}).reindex(ret_pivot.columns, fill_value=0)).values.tolist()

    eta = float(profile.get('eta', 1.0))
    current_z = calculate_current_z(float(portfolio.get('last_z', 1.0)), daily_rets, eta)

    days_since = len([d for d in get_trading_days_range(pd.to_datetime(last_rec['rebalance_date']).date(), int((current_date - pd.to_datetime(last_rec['rebalance_date']).date()).days * 1.5)) if pd.to_datetime(last_rec['rebalance_date']).date() < d.date() <= current_date])

    session = db.get_session(st.session_state.session_id)
    criteria_met, triggers = check_rebalance_criteria(
        last_rec.get('discrete_holdings', {}), last_rec.get('theoretical_weights', {}), market_data,
        current_z, float(portfolio['last_z']), days_since,
        float(session['rebalance_policy_weights']),
        float(session['rebalance_policy_benchmark']),
        int(session['rebalance_policy_planned'])
    )

    open_date = pd.to_datetime(portfolio['created_at']).date()
    horizon_days = int(portfolio.get('horizon_days', 60))
    close_dates = get_trading_days_range(open_date, horizon_days)
    close_date = close_dates[-1].date() if not close_dates.empty else open_date + timedelta(days=horizon_days)

    st.session_state['portfolio_current_z'] = current_z
    st.session_state['portfolio_close_date'] = close_date
    st.session_state['portfolio_triggers'] = triggers if criteria_met else []

    if not criteria_met and not st.session_state.get('force_rebalance'):
        return

    st.session_state['pending_rebalance_data'] = {
        "discrete_holdings": last_rec.get('discrete_holdings', {}),
        "cash": last_rec.get('cash', 0.0),
        "date": current_date,
        "weights": last_rec.get('theoretical_weights', {}),
        "current_z": current_z
    }
    st.rerun()


@st.dialog("Вход в пользовательскую сессию", width="large", dismissible=False)
def show_session_modal(db: DatabaseManager) -> None:
    """Отображает модальное окно для входа или создания сессии.
    
    Args:
        db: Экземпляр DatabaseManager
    """
    from .data_cache import profiles_df
    from .date_utils import get_active_date
    if st.session_state.get('session_id'):
        return
    mode = st.segmented_control(
        "",
        ["Войти в существующую сессию", "Создать новую сессию"],
        default="Войти в существующую сессию"
    )

    if mode == "Войти в существующую сессию":
        sid_input = st.text_input("ID сессии", placeholder="00000000-0000-0000-0000-000000000000")
        if st.button("Подтвердить вход", type="primary", width='stretch'):
            if not sid_input.strip():
                st.error("Введите ID сессии")
                return
            try:
                row = db.get_session(sid_input)
                if row:
                    st.session_state.session_id = row['session_id']
                    st.session_state.risk_profile_id = row['profile_id']
                    st.rerun()
                else:
                    st.error("Сессия не найдена")
            except Exception as e:
                st.error(f"Ошибка ввода")
    else:
        st.header('Выберите профиль риска')

        if 'dialog_selected_profile' not in st.session_state:
            st.session_state.dialog_selected_profile = None

        cards_data = []
        for _, row in profiles_df.iterrows():
            cards_data.append({
                "title": row['display_name'],
                "description": row["description"],
                "icon": f":material/{row['icon']}:",
                "id": row['profile_id']
            })

        selected_idx = card_selector(cards_data)
        if selected_idx is not None:
            st.session_state.dialog_selected_profile = selected_idx

        if st.session_state.dialog_selected_profile is not None:
            st.header('Пользовательская настройка')
            selected_profile = profiles_df.iloc[st.session_state.dialog_selected_profile]

            initial_balance = st.number_input(
                    "Начальный капитал (RUB)",
                    value=100000.00,
                    step=10000.00,
                    help="Величина начального капитала"
                )

            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                rebalance_policy_planned = st.number_input(
                    "Частота ребалансировки (торговые дни)",
                    value=selected_profile["rebalance_policy_planned"],
                    step=1,
                    help="Определяет частоту (в торговых днях), с которой будет проводиться плановая ребалансировка"
                )
            with col_2:
                rebalance_policy_benchmark = st.number_input(
                    "Допустимое отклонение бенчмарка (%)",
                    value=float(selected_profile["rebalance_policy_benchmark"] * 100),
                    step=1.0,
                    help="Определяет минимальное отклонение бенчмарка от предыдущего значения, при котором будет проводиться ребалансировка"
                )
            with col_3:
                rebalance_policy_weights = st.number_input(
                    "Допустимое отклонение структуры (%)",
                    value=float(selected_profile["rebalance_policy_weights"] * 100),
                    step=1.0,
                    help="Определяет минимальное отклонение структуры портфеля от рекомендуемого, при котором будет проводиться ребалансировка"
                )

        if st.button("Создать сессию", type="primary"):
            profile_id = selected_profile['profile_id']
            session_id = db.create_session(
                profile_id=profile_id,
                balance=str(initial_balance),
                rebalance_policy_planned=str(rebalance_policy_planned),
                rebalance_policy_benchmark=str(rebalance_policy_benchmark/100),
                rebalance_policy_weights=str(rebalance_policy_weights/100),
                created_at=get_active_date()
            )
            st.session_state.session_id = session_id
            st.session_state.risk_profile_id = profile_id
            st.rerun()


def initialize_session() -> str:
    """
    Инициализирует пользовательскую сессию приложения.
    
    Returns:
        id текущей сессии
    """
    from .data_cache import db
    from .date_utils import get_active_date
    if st.session_state.get('_session_initialized'):
        return st.session_state.session_id
    if 'session_id' not in st.session_state:
        show_session_modal(db)
        st.stop()
    st.session_state._session_initialized = True

    session_id = st.session_state.session_id
    if APP_MODE == "demo" and st.session_state.get("demo_date_override") is None:
        st.session_state.demo_date_override = get_active_date()

    active_p = db.get_active_portfolio(session_id)
    if active_p:
        st.session_state.portfolio_id = active_p['portfolio_id']
        _check_portfolio_triggers(active_p['portfolio_id'])
    return session_id


def exit_session() -> None:
    """Завершает текущую сессию и очищает состояние."""
    from .data_cache import db
    keys_to_clear = [
        'session_id', 'risk_profile_id', 'portfolio_id', 'portfolio', 'backtest_results',
        'dialog_selected_profile', '_session_initialized', 'last_tickers_tab3',
        'benchmark_allocations', 'sma_windows', 'portfolio_config',
        'temp_portfolio', 'temp_horizon', 'pending_creation',
        'demo_date_override', 'pending_rebalance_data', 'force_rebalance',
        'portfolio_current_z', 'portfolio_close_date', 'portfolio_triggers',
        'last_criteria_check_date', 'criteria_checked_date'
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


def reset_session() -> None:
    """Сбрасывает сессию и деактивирует текущий сеанс в БД."""
    from .data_cache import db
    current_sid = st.session_state.get('session_id')
    if current_sid:
        db.deactivate_session(current_sid)

    keys_to_clear = [
        'session_id', 'risk_profile_id', 'portfolio_id', '_session_initialized',
        'demo_date_override', 'dialog_selected_profile',
        'pending_creation', 'temp_recommendation',
        'last_criteria_check_date', 'criteria_checked_date',
        'portfolio_current_z', 'portfolio_close_date', 'portfolio_triggers',
        'pending_rebalance_data', 'force_rebalance',
        'last_tickers_tab3', 'benchmark_allocations', 'sma_windows',
        'portfolio_config', 'temp_portfolio', 'temp_horizon',
        'backtest_results', 'screener_results'
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.cache_data.clear()
    st.cache_resource.clear()

    st.rerun()

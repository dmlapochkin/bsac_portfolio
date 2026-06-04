"""Модуль управления портфелем и ребалансировкой."""

from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

from bsac_portfolio.bsac.drl_inference import predict_weights
from bsac_portfolio.database.db_manager import DatabaseManager
from bsac_portfolio.portfolio.portfolio_logic import (
    calculate_forecast_metrics,
    calculate_rebalance_actions,
    discretize_portfolio,
    run_monte_carlo_forecast,
)


def render_portfolio_tab(db: DatabaseManager) -> None:
    """
    Отображает вкладку активного портфеля.
    
    Args:
        db: Экземпляр DatabaseManager
    """
    from .date_utils import get_active_date, _fill_portfolio_history
    from .portfolio_dashboard import render_portfolio_dashboard
    session_id = st.session_state.get("session_id")
    if not session_id:
        st.warning("Сессия не активна")
        return
    active_portfolio = db.get_active_portfolio(session_id)
    if not active_portfolio:
        render_portfolio_creation(db)
        return

    portfolio_id = active_portfolio["portfolio_id"]
    _fill_portfolio_history(portfolio_id, get_active_date())
    history = db.get_rebalance_history_detailed(portfolio_id)
    executed = [h for h in history if h.get("status") == "executed"]

    pending_db = db.get_pending_rebalance(portfolio_id)
    is_expired = pending_db and pd.to_datetime(pending_db["rebalance_date"]).date() < get_active_date()
    if is_expired:
        db.update_rebalance_status(pending_db["id"], "rejected")
        pending_db = None
        st.session_state.pop("pending_rebalance_data", None)

    pending = pending_db.copy() if pending_db else {}
    if "pending_rebalance_data" in st.session_state:
        pending.update(st.session_state.pending_rebalance_data)
    pending = pending if pending else None

    render_portfolio_dashboard(db, active_portfolio, executed, pending)


def render_portfolio_creation(db: DatabaseManager) -> None:
    """
    Отображает интерфейс создания нового портфеля.
    
    Args:
        db: Экземпляр DatabaseManager
    """
    from .config import MODEL_BASE_DIR, COMMISSION_RATE, MIN_POSITION_WEIGHT, load_model_metadata, format_money
    from .data_cache import _get_cached_agent
    from .date_utils import get_active_date, is_trading_day, get_trading_days_range
    from .ui_components import render_portfolio_structure_ui, render_forecast_plot
    profile = db.get_risk_profile(st.session_state.risk_profile_id)
    if not profile or not profile.get("model_name"):
        st.error("Профиль риска или имя модели не настроены")
        return
    session_data = db.get_session(st.session_state.session_id)
    current_balance = float(session_data['current_balance'])
    model_dir = f"{MODEL_BASE_DIR}/{profile['model_name']}"
    meta = load_model_metadata(model_dir)
    if not meta:
        st.error(f"Модель '{model_dir}' не найдена или повреждена.")
        return

    env_cfg = meta.get("env_config", {})
    tickers = env_cfg.get("tickers", [])
    feature_cols = env_cfg.get("feature_columns", [])
    window_size = env_cfg.get("window_size", 60)

    chosen_tickers = st.multiselect("Активы", tickers, tickers, disabled=True)

    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        horizon = st.number_input("Инвестиционный горизонт (торговых дней)", min_value=1, max_value=500, value=60, step=1, key="new_hor")
    with col_2:
        open_date = get_active_date()
        st.metric("Дата открытия", open_date.strftime("%d.%m.%Y"))
    with col_3:
        close_dates = get_trading_days_range(open_date, horizon)
        close_date = close_dates[-1].date() if len(close_dates) > 0 else open_date + timedelta(days=horizon)
        st.metric("Дата закрытия", close_date.strftime("%d.%m.%Y"))

    if not is_trading_day(open_date):
        st.warning("Выбранная дата не является торговым днем")
        st.stop()

    if st.button("Сформировать портфель", width='stretch', type="primary", key="calc_new"):
        st.session_state.pending_creation = {
            "balance": current_balance, "horizon": horizon, "model_dir": model_dir,
            "tickers": chosen_tickers, "feature_cols": feature_cols, "window_size": window_size
        }
        st.rerun()

    if "pending_creation" in st.session_state:
        cfg = st.session_state.pop("pending_creation")
        with st.spinner("Формирование портфеля"):
            weights = predict_weights(
                agent=_get_cached_agent(cfg["model_dir"]), db=db, tickers=cfg["tickers"],
                model_dir=cfg["model_dir"], feature_cols=cfg["feature_cols"],
                window_size=cfg["window_size"], reference_date=get_active_date()
            )
            market_data = db.get_latest_market_data(list(weights.keys()), target_date=get_active_date())
            discrete, rem, comm_amount = discretize_portfolio(weights, market_data, cfg["balance"], COMMISSION_RATE, previous_holdings=None, min_weight=MIN_POSITION_WEIGHT)
            st.session_state.temp_recommendation = {
                "weights": weights, "discrete": discrete,
                "remaining": rem, "balance": cfg["balance"], "horizon": cfg["horizon"], "commission": comm_amount
            }
        st.rerun()

    if "temp_recommendation" in st.session_state:
        st.space()
        temp = st.session_state.temp_recommendation
        open_date = get_active_date()
        close_dates = get_trading_days_range(open_date, temp['horizon'])
        close_date = close_dates[-1].date() if len(close_dates) > 0 else open_date + timedelta(days=temp['horizon'])
        st.header("Структура портфеля")
        tickers = list(temp["discrete"].keys())
        market_data = db.get_latest_market_data(tickers, target_date=get_active_date())
        cash = temp["remaining"]
        col_21, col_22, col_23 = st.columns(3)
        col_21.metric("Стоимость активов (RUB)", f"{format_money(temp['balance']-cash)}")
        col_22.metric("Свободные средства (RUB)", f"{format_money(cash)}")
        col_23.metric("Комиссия (RUB)", f"{format_money(temp['commission'])}")
        render_portfolio_structure_ui(temp["discrete"], market_data, key="render_portfolio_creation")
        st.space()
        st.header("Прогноз и оценка рисков")
        hist = db.get_market_data(tickers, (get_active_date() - timedelta(days=365)).strftime("%Y-%m-%d"), get_active_date().strftime("%Y-%m-%d"))
        if not hist.empty and "close" in hist.columns:
            hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
            hist["log_ret"] = np.log(hist["close"] / hist.groupby("ticker")["close"].shift(1))
            ret_pivot = hist.pivot(index="date", columns="ticker", values="log_ret").dropna()
            mc = run_monte_carlo_forecast(temp["discrete"], market_data, ret_pivot, horizon_days=temp['horizon'])
            if mc and mc.get("cum_paths") is not None:
                metrics = calculate_forecast_metrics(mc["cum_paths"], mc["start_val"], temp["horizon"])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ожид. стоимость активов (RUB)", f"{format_money(mc['p50'][-1])}")
                c2.metric("Ожид. доходность", f"{metrics['expected_return']:.2%}")
                c3.metric("Годовая доходность", f"{metrics['expected_annual_return']:.2%}")
                c4.metric("Волатильность (год.)", f"{metrics['volatility']:.2%}")
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Коэф. Шарпа", f"{metrics['sharpe']:.2f}")
                c6.metric("Коэф. Сортино", f"{metrics['sortino']:.2f}")
                c7.metric("VaR (95%)", f"{metrics['var_95']:.2%}")
                c8.metric("CVaR (95%)", f"{metrics['cvar_95']:.2%}")
                render_forecast_plot(mc, cash, open_date, temp["horizon"])
            else:
                st.warning("Недостаточно данных для симуляции")
        else:
            st.warning("Недостаточно исторических данных для оценки рисков")
        st.space()
        col_acc, col_rej = st.columns(2)
        with col_acc:
            if st.button("Принять и создать портфель", type="primary", use_container_width=True):
                try:
                    comm_display = float(temp["commission"]) if "commission" in temp else float(temp["balance"]) * COMMISSION_RATE
                    portfolio_id = db.create_portfolio(session_id=st.session_state.session_id, initial_balance=float(temp["balance"]), horizon_days=int(temp["horizon"]), created_at=get_active_date(), commission=comm_display)
                    st.session_state.portfolio_id = portfolio_id
                    db.save_rebalance_record(portfolio_id=portfolio_id, date=get_active_date().strftime("%Y-%m-%d"), theoretical_weights=temp["weights"], status='executed', cash=float(temp["remaining"]), commission=comm_display, discrete_holdings=temp["discrete"])
                    db.save_portfolio_snapshot(portfolio_id, open_date, float(temp["remaining"]), float(temp["balance"]) - comm_display, 1.0, temp["discrete"])
                    st.session_state.pop("temp_recommendation", None)
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка: {e}")
        with col_rej:
            if st.button("Отклонить", use_container_width=True):
                st.session_state.pop("temp_recommendation", None)
                st.rerun()


def render_rebalance_panel(db: DatabaseManager, portfolio: dict, pending: dict, executed: list, reason: str = "") -> None:
    """
    Отображает панель ребалансировки портфеля.
    
    Args:
        db: Экземпляр DatabaseManager
        portfolio: Словарь с данными портфеля
        pending: Данные по текущей доступной ребалансировке
        executed: Список исполненных ребалансировок
        reason: Причина ребалансировки
    """
    from .config import format_money
    from .date_utils import get_active_date, get_trading_days_range
    from .ui_components import render_portfolio_structure_ui, render_forecast_plot
    proposed_holdings = pending.get("discrete_holdings", {})
    proposed_cash = float(pending.get("cash", 0.0)) if pending.get("cash") is not None else 0.0
    commission = float(pending.get("commission", 0.0)) if pending.get("commission") is not None else 0.0
    tickers = list(proposed_holdings.keys())
    market_data = db.get_latest_market_data(tickers, target_date=get_active_date())

    if reason:
        for item in reason.split(", "):
            st.badge(item.capitalize(), icon=":material/warning:", color="green")

    if tickers:
        open_date = pd.to_datetime(portfolio["created_at"]).date()
        current_date = get_active_date()
        horizon_days = int(portfolio.get("horizon_days", 60))
        all_planned_days = get_trading_days_range(open_date, horizon_days)
        elapsed_days = len([d for d in all_planned_days if d.date() <= current_date])
        remaining_days = max(1, horizon_days - elapsed_days)
        invested_value = sum(l * market_data.get(t, {}).get("close", 0) * market_data.get(t, {}).get("lot_size", 1) for t, l in proposed_holdings.items())

        total_portfolio_value = invested_value + proposed_cash
        prev_holdings = executed[-1].get("discrete_holdings", {}) if executed else {}
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Общий капитал", f"{format_money(total_portfolio_value)}")
        r2.metric("Стоимость активов", f"{format_money(invested_value)}")
        r3.metric("Свободный остаток", f"{format_money(proposed_cash)}")
        r4.metric("Комиссия", f"{format_money(commission)}")
        actions_df, turnover = calculate_rebalance_actions(proposed_holdings, prev_holdings, market_data)

        if not actions_df.empty:
            if "Цена" in actions_df.columns:
                actions_df["Цена"] = actions_df["Цена"].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "0.00")
            if "Стоимость" in actions_df.columns:
                actions_df["Стоимость"] = actions_df["Стоимость"].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "0.00")
            st.dataframe(actions_df, width='stretch', hide_index=True, column_config={"Цена": st.column_config.TextColumn(alignment="right"), "Стоимость": st.column_config.TextColumn(alignment="right")})
    else:
        st.warning("Рекомендация не содержит позиций")

    st.header("Структура портфеля")
    render_portfolio_structure_ui(proposed_holdings, market_data, key="render_rebalance_panel")

    st.header("Прогноз и оценка рисков")
    mc = None
    forecast_metrics = {"expected_return": 0.0, "volatility": 0.0, "sharpe": 0.0, "sortino": 0.0, "var_95": 0.0, "cvar_95": 0.0}
    hist_ret = db.get_market_data(tickers, (current_date - timedelta(days=365)).strftime("%Y-%m-%d"), current_date.strftime("%Y-%m-%d"))
    if not hist_ret.empty and "close" in hist_ret.columns:
        hist_ret["close"] = pd.to_numeric(hist_ret["close"], errors="coerce")
        hist_ret = hist_ret.sort_values(['ticker', 'date'])
        hist_ret["log_ret"] = hist_ret.groupby("ticker")["close"].pct_change()
        ret_pivot = hist_ret.pivot(index="date", columns="ticker", values="log_ret").dropna()
        mc = run_monte_carlo_forecast(proposed_holdings, market_data, ret_pivot, horizon_days=remaining_days, n_simulations=200)
        if mc:
            forecast_metrics = calculate_forecast_metrics(mc["cum_paths"], invested_value, remaining_days)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ожид. стоимость (активы)", f"{format_money(mc['p50'][-1])}")
    c2.metric("Ожид. доходность", f"{forecast_metrics['expected_return']:.2%}")
    c3.metric("Годовая доходность", f"{forecast_metrics['expected_annual_return']:.2%}")
    c4.metric("Волатильность (год.)", f"{forecast_metrics['volatility']:.2%}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Коэф. Шарпа", f"{forecast_metrics['sharpe']:.2f}")
    c6.metric("Коэф. Сортино", f"{forecast_metrics['sortino']:.2f}")
    c7.metric("VaR (95%)", f"{forecast_metrics['var_95']:.2%}")
    c8.metric("CVaR (95%)", f"{forecast_metrics['cvar_95']:.2%}")

    if mc:
        render_forecast_plot(mc, proposed_cash, current_date, remaining_days)
    else:
        st.warning("Недостаточно данных для прогноза")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Принять ребалансировку", type="primary", width='stretch'):
            db.update_rebalance_status(pending["id"], "executed")

            proposed_holdings = pending.get("discrete_holdings", {})
            proposed_cash = float(pending.get("cash", 0.0)) if pending.get("cash") is not None else 0.0
            tickers = list(proposed_holdings.keys())
            market_data = db.get_latest_market_data(tickers, target_date=get_active_date())

            invested_value = sum(l * market_data.get(t, {}).get("close", 0) * market_data.get(t, {}).get("lot_size", 1) for t, l in proposed_holdings.items())
            total_portfolio_value = invested_value + proposed_cash

            db.save_portfolio_snapshot(
                portfolio["portfolio_id"],
                get_active_date(),
                proposed_cash,
                total_portfolio_value,
                pending.get("current_z", 1.0),
                proposed_holdings
            )

            db.update_current_balance(portfolio["portfolio_id"], st.session_state.session_id, total_portfolio_value)

            st.session_state.pop("pending_rebalance_data", None)
            st.session_state.pop("last_criteria_check_date", None)
            st.rerun()
    with c2:
        if st.button("Отклонить", width='stretch'):
            db.update_rebalance_status(pending["id"], "rejected")
            st.session_state.pop("pending_rebalance_data", None)
            st.rerun()

"""Модуль дашборда портфеля и генерации рекомендаций."""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bsac_portfolio.bsac.drl_inference import predict_weights
from bsac_portfolio.database.db_manager import DatabaseManager
from bsac_portfolio.portfolio.portfolio_logic import (
    calculate_current_z,
    calculate_portfolio_metrics,
    check_rebalance_criteria,
    discretize_portfolio,
)
from .portfolio_manager import render_rebalance_panel
from .ui_components import render_rebalance_operations


def render_portfolio_dashboard(db: DatabaseManager, portfolio: dict, executed: list, pending: dict = None, hide_controls: bool = False) -> None:
    """
    Отображает дашборд активного портфеля с метриками и графиками.
    
    Args:
        db: Экземпляр DatabaseManager
        portfolio: Словарь с данными портфеля
        executed: Список исполненных ребалансировок
        pending: Данные по текущей доступной ребалансировке
        hide_controls: Флаг для скрытия кнопок управления
    """
    from .config import format_money
    from .date_utils import get_active_date, get_trading_days_range, _fill_portfolio_history
    from .ui_components import render_portfolio_structure_ui, render_rebalance_operations
    portfolio_id = portfolio["portfolio_id"]
    current_date = get_active_date()

    if executed and portfolio.get("status") != "closed" and st.session_state.get("last_criteria_check_date") != current_date:
        last_rec = executed[0]
        tickers = list(last_rec.get("discrete_holdings", {}).keys())
        market_data = db.get_latest_market_data(tickers, target_date=current_date)

        current_val = float(last_rec.get("cash", 0.0))
        for t, lots in last_rec.get("discrete_holdings", {}).items():
            info = market_data.get(t)
            if info:
                current_val += lots * info["close"] * info["lot_size"]

        hist_returns = db.get_market_data(tickers, pd.to_datetime(last_rec["rebalance_date"]).date().strftime("%Y-%m-%d"), current_date.strftime("%Y-%m-%d"))
        daily_rets = []
        if not hist_returns.empty and "close" in hist_returns.columns:
            ret_pivot = hist_returns.pivot(index='date', columns='ticker', values='close').astype(float).pct_change().dropna()
            weights = last_rec.get("theoretical_weights", {})
            daily_rets = ret_pivot.dot(pd.Series({k: float(v) for k, v in weights.items()}).reindex(ret_pivot.columns, fill_value=0)).values.tolist()

        profile = db.get_risk_profile(st.session_state.risk_profile_id)
        eta = float(profile.get("eta", 1.0)) if profile else 1.0
        current_z = calculate_current_z(float(portfolio.get("last_z", 1.0)), daily_rets, eta)

        days_since = len([d for d in get_trading_days_range(pd.to_datetime(last_rec["rebalance_date"]).date(), int((current_date - pd.to_datetime(last_rec["rebalance_date"]).date()).days * 1.5)) if pd.to_datetime(last_rec["rebalance_date"]).date() < d.date() <= current_date])

        session = db.get_session(st.session_state.session_id)
        criteria_met, triggers = check_rebalance_criteria(
            last_rec.get("discrete_holdings", {}), last_rec.get("theoretical_weights", {}), market_data,
            current_z, float(portfolio["last_z"]), days_since,
            float(session["rebalance_policy_weights"]),
            float(session["rebalance_policy_benchmark"]),
            int(session["rebalance_policy_planned"])
        )

        st.session_state["portfolio_current_z"] = current_z
        st.session_state["portfolio_triggers"] = triggers if criteria_met else []
        st.session_state["last_criteria_check_date"] = current_date

        if criteria_met and not st.session_state.get("pending_rebalance_data"):
            generate_rebalance_recommendation(db, st.session_state.session_id, portfolio_id, float(portfolio["initial_balance"]), force=False)

    _fill_portfolio_history(portfolio_id, current_date)
    snap = db.get_portfolio_snapshot(portfolio_id, current_date)
    if snap:
        current_holdings = snap.get("holdings", {})
        last_cash = float(snap["cash"])
        current_total_value = float(snap["portfolio_value"])
    else:
        current_holdings = executed[-1].get("discrete_holdings", {}) if executed else {}
        last_cash = float(executed[-1].get("cash", 0.0)) if executed and executed[-1].get("cash") is not None else 0.0
        tickers = list(current_holdings.keys())
        market_data = db.get_latest_market_data(tickers, target_date=current_date) if tickers else {}
        invested_value = sum(l * market_data.get(t, {}).get("close", 0) * market_data.get(t, {}).get("lot_size", 1) for t, l in current_holdings.items())
        current_total_value = last_cash + invested_value

    if not hide_controls:
        db.update_current_balance(portfolio_id, st.session_state.session_id, current_total_value)

    tickers = list(current_holdings.keys())
    market_data = db.get_latest_market_data(tickers, target_date=current_date) if tickers else {}
    initial_balance = float(portfolio["initial_balance"])
    open_date = pd.to_datetime(portfolio.get("created_at", current_date)).date()
    horizon_trading_days = int(portfolio.get("horizon_days", 60))
    close_dates = get_trading_days_range(open_date, horizon_trading_days)
    close_date = close_dates[-1].date() if len(close_dates) > 0 else open_date + timedelta(days=horizon_trading_days)

    is_closed = portfolio.get("status") == "closed"
    is_past_horizon = current_date >= close_date

    if is_closed:
        st.session_state.pop("pending_rebalance_data", None)
        st.session_state.pop("portfolio_triggers", None)
        st.info("Портфель закрыт")
    elif is_past_horizon:
        st.warning("Инвестиционный период завершен")

    if pending and not is_closed and not is_past_horizon:
        st.toast("Доступна ребалансировка", duration="long")
        with st.expander("**Доступна ребалансировка**", expanded=False):
            render_rebalance_panel(db, portfolio, pending, executed, pending.get("reason", ""))
        st.space()

    current_total_return = (current_total_value / initial_balance - 1.0) if initial_balance > 0 else 0.0

    lot_sizes = {t: info["lot_size"] for t, info in market_data.items()}
    metrics = {}
    snap_hist = db.get_portfolio_history(portfolio_id, open_date, current_date)
    has_history = len(snap_hist) >= 2
    if has_history:
        values = [float(s['portfolio_value']) for s in snap_hist]
        values_series = pd.Series(values)
        returns = values_series.pct_change().dropna()

        if len(returns) >= 1:
            total_return = values[-1] / initial_balance - 1.0
            n_days = len(returns)
            ann_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
            sharpe = ann_return / volatility if volatility > 0 else 0.0

            cum = (1 + returns).cumprod()
            max_dd = (cum / cum.cummax() - 1).min()
            var_95 = np.percentile(returns, 5)
            tail = returns[returns <= var_95]
            cvar_95 = tail.mean() if not tail.empty else var_95

            metrics = {
                'total_return': total_return, 'annual_return': ann_return, 'volatility': volatility, 'sharpe': sharpe,
                'max_drawdown': max_dd, 'var_95': var_95, 'cvar_95': cvar_95,
                'sortino': (ann_return / (np.std(returns[returns < 0]) * np.sqrt(252)) if len(returns[returns < 0]) > 1 else 0.0)
            }
    elif has_history:
        hist_data = db.get_market_data(tickers, executed[0]["rebalance_date"], current_date.strftime("%Y-%m-%d"))
        lot_sizes = {t: info["lot_size"] for t, info in market_data.items()}
        metrics = calculate_portfolio_metrics(executed, hist_data, initial_balance, lot_sizes)

    c1, c2, c3 = st.columns(3)
    c1.metric("Общий капитал (RUB)", f"{format_money(current_total_value)}", delta=f"{current_total_return:.2%}")
    c2.metric("Стоимость активов (RUB)", f"{format_money(current_total_value - last_cash)}")
    c3.metric("Свободный остаток (RUB)", f"{format_money(last_cash)}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Дата открытия", open_date.strftime("%d.%m.%Y"))
    c5.metric("Дата закрытия", close_date.strftime("%d.%m.%Y"))
    c6.metric("Горизонт (торг. дни)", f"{horizon_trading_days} дн.")

    st.space()
    st.header("Оценка эффективности")
    if has_history:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Доходность", f"{metrics.get('total_return', 0):.2%}")
        m2.metric("Годовая доходность", f"{metrics.get('annual_return', 0):.2%}")
        m3.metric("Волатильность (год.)", f"{metrics.get('volatility', 0):.2%}")
        m4.metric("Макс. просадка", f"{metrics.get('max_drawdown', 0):.2%}")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Коэф. Шарпа", f"{metrics.get('sharpe', 0):.2f}")
        m6.metric("Коэф. Сортино", f"{metrics.get('sortino', 0):.2f}")
        m7.metric("VaR (95%)", f"{metrics.get('var_95', 0):.2%}")
        m8.metric("CVaR (95%)", f"{metrics.get('cvar_95', 0):.2%}")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Доходность", "–")
        m2.metric("Годовая доходность", "–")
        m3.metric("Волатильность (год.)", "–")
        m4.metric("Макс. просадка", "–")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Коэф. Шарпа", "–")
        m6.metric("Коэф. Сортино", "–")
        m7.metric("VaR (95%)", "–")
        m8.metric("CVaR (95%)", "–")

    if has_history:
        render_portfolio_charts(db, portfolio, executed, current_holdings, market_data, last_cash, initial_balance, horizon_trading_days)

    st.space()
    st.header("Структура портфеля")

    render_portfolio_structure_ui(current_holdings, market_data, current_total_value, key="render_portfolio_dashboard")
    st.session_state['criteria_checked_date'] = current_date

    with st.expander("История операций", expanded=False):
        if executed:
            chron_executed = sorted(executed, key=lambda x: x['rebalance_date'])
            for i, rec in enumerate(chron_executed):
                prev_holdings = chron_executed[i-1].get("discrete_holdings", {}) if i > 0 else {}
                text = "Ребалансировка" if prev_holdings else "Формирование портфеля"
                reason_display = f"( ({rec.get('reason')})" if i > 0 and rec.get('reason') else ""
                with st.expander(f"{rec['rebalance_date'].strftime('%d.%m.%Y')} – {text}{reason_display}", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Стоимость активов (RUB)", format_money(rec.get("portfolio_value") or 0))
                    c2.metric("Свободный остаток (RUB)", format_money(rec.get("cash", 0) or 0))
                    c3.metric("Комиссия (RUB)", format_money(rec.get("commission", 0) or 0))
                    render_rebalance_operations(db, rec, prev_holdings)
        else:
            st.info("Операции еще не проводились")

    if not hide_controls:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("Сформировать новую рекомендацию", width='stretch', type="secondary", disabled=is_closed or is_past_horizon):
                generate_rebalance_recommendation(db, st.session_state.session_id, portfolio_id, initial_balance, force=True)
        with col_c2:
            close_label = "Закрыть портфель" if not is_closed else "Портфель закрыт"
            if st.button(close_label, type="primary", width='stretch', disabled=is_closed):
                db.update_current_balance(portfolio_id, st.session_state.session_id, current_total_value)
                db.close_portfolio(portfolio_id, current_date)
                st.session_state.pop("pending_rebalance_data", None)
                st.session_state.pop("portfolio_triggers", None)
                st.session_state.portfolio_id = None
                st.rerun()


def generate_rebalance_recommendation(db: DatabaseManager, session_id: str, portfolio_id: str, initial_balance: float, force: bool = False) -> bool:
    """
    Генерирует рекомендацию по ребалансировке портфеля.
    
    Args:
        db: Экземпляр DatabaseManager
        session_id: id сессии
        portfolio_id: id портфеля
        initial_balance: Начальный баланс портфеля
        force: Флаг принудительной генерации

    Returns:
        Флаг успешности генерации рекомендации
    """
    from .config import MODEL_BASE_DIR, COMMISSION_RATE, MIN_POSITION_WEIGHT, load_model_metadata
    from .data_cache import _get_cached_agent
    from .date_utils import get_active_date, get_trading_days_range
    try:
        session = db.get_session(session_id)
        history = db.get_rebalance_history_detailed(portfolio_id)
        executed = [h for h in history if h.get("status") == "executed"]
        if not executed:
            st.error("Нет исполненных ребалансировок для анализа")
            return False

        last_rec = executed[-1]
        last_theoretical = last_rec["theoretical_weights"]
        latest_state = db.get_latest_portfolio_state(portfolio_id)
        if latest_state and latest_state.get("holdings"):
            last_holdings = latest_state["holdings"]
            last_cash = float(latest_state.get("cash", 0.0))
        else:
            last_holdings = last_rec.get("discrete_holdings", {})
            last_cash = float(last_rec.get("cash", 0.0))

        profile = db.get_risk_profile(st.session_state.risk_profile_id)
        if not profile:
            st.error("Профиль риска не найден")
            return False

        model_dir = f"{MODEL_BASE_DIR}/{profile['model_name']}"
        meta = load_model_metadata(model_dir)
        if not meta:
            st.error(f"Модель '{profile['model_name']}' не найдена.")
            return False

        env_cfg = meta.get("env_config", {})
        md_current = db.get_latest_market_data(list(last_holdings.keys()), target_date=get_active_date())
        current_val = last_cash
        for t, lots in last_holdings.items():
            info = md_current.get(t)
            if info:
                current_val += lots * info["close"] * info["lot_size"]

        current_date = get_active_date()
        portfolio_obj = db.get_portfolio(portfolio_id)
        last_z = float(portfolio_obj.get("last_z", 1.0))
        days_since = len([d for d in get_trading_days_range(pd.to_datetime(last_rec["rebalance_date"]).date(), int((current_date - pd.to_datetime(last_rec["rebalance_date"]).date()).days * 1.5)) if d.date() > pd.to_datetime(last_rec["rebalance_date"]).date() and d.date() <= current_date])

        hist_returns = db.get_market_data(list(last_theoretical.keys()), pd.to_datetime(last_rec["rebalance_date"]).date().strftime("%Y-%m-%d"), current_date.strftime("%Y-%m-%d"))
        daily_rets = []
        if not hist_returns.empty and "close" in hist_returns.columns:
            hist_returns["close"] = pd.to_numeric(hist_returns["close"], errors="coerce")
            ret_pivot = hist_returns.pivot(index="date", columns="ticker", values="close").pct_change().dropna()
            port_vals = ret_pivot.dot(pd.Series({k: float(v) for k, v in last_theoretical.items()}).reindex(ret_pivot.columns, fill_value=0))
            daily_rets = port_vals.values.tolist()

        current_z = calculate_current_z(last_z, daily_rets, float(profile.get("eta", 1.0)))

        reason_str = ""
        if not force:
            criteria_met, triggers = check_rebalance_criteria(
                last_holdings, last_theoretical, md_current,
                current_z, last_z, days_since,
                float(session["rebalance_policy_weights"]),
                float(session["rebalance_policy_benchmark"]),
                int(session["rebalance_policy_planned"])
            )
            reason_str = ", ".join(triggers) if triggers else "Автоматическая проверка"
            reason_str = reason_str.capitalize()
        else:
            reason_str = "Ручная ребалансировка"

        pending_today = db.get_pending_rebalance(portfolio_id)
        if pending_today and pd.to_datetime(pending_today["rebalance_date"]).date() == current_date and not force:
            st.session_state.pending_rebalance_data = {
                "discrete_holdings": pending_today.get("discrete_holdings", {}),
                "cash": float(pending_today.get("cash", 0.0)),
                "commission": float(pending_today.get("commission", 0.0)),
                "date": current_date,
                "weights": pending_today.get("theoretical_weights", {}),
                "current_z": current_z
            }
            st.rerun()
            return True

        weights = predict_weights(
            agent=_get_cached_agent(model_dir), db=db,
            tickers=env_cfg.get("tickers", []), model_dir=model_dir,
            feature_cols=env_cfg.get("feature_columns", []),
            window_size=env_cfg.get("window_size", 60),
            reference_date=get_active_date()
        )

        md_new = db.get_latest_market_data(list(weights.keys()), target_date=get_active_date())
        prev_holdings = last_holdings if last_holdings else {}
        discrete, rem, comm_amount = discretize_portfolio(
            weights, md_new, current_val,
            COMMISSION_RATE, previous_holdings=last_holdings, min_weight=MIN_POSITION_WEIGHT
        )

        db.update_z(portfolio_id, current_z)
        db.save_rebalance_record(portfolio_id, current_date.strftime("%Y-%m-%d"), weights, status='pending', cash=rem, reason=reason_str, commission=comm_amount, discrete_holdings=discrete)

        st.session_state.pending_rebalance_data = {
            "discrete_holdings": discrete,
            "cash": rem,
            "commission": comm_amount,
            "date": current_date,
            "weights": weights,
            "current_z": current_z
        }
        st.rerun()
        return True
    except Exception as e:
        return False


def render_portfolio_charts(db: DatabaseManager, portfolio: dict, executed: list, current_holdings: dict, market_data: dict, cash: float, initial_balance: float, horizon_trading_days: int) -> None:
    """
    Отображает график стоимости портфеля.
    
    Args:
        db: Экземпляр DatabaseManager
        portfolio: Словарь с данными портфеля
        executed: Список исполненных ребалансировок
        current_holdings: Текущие позиции в портфеле
        market_data: Словарь с рыночными данными по тикерам
        cash: Текущий свободный остаток
        initial_balance: Начальный баланс портфеля
        horizon_trading_days: Количество торговых дней в инвестиционном горизонте
    """
    from .date_utils import get_active_date
    portfolio_id = portfolio["portfolio_id"]
    open_date = pd.to_datetime(portfolio.get("created_at", get_active_date()))
    snap_hist = db.get_portfolio_history(portfolio_id, open_date, get_active_date())

    dates, values = [], []
    if snap_hist:
        for s in snap_hist:
            dates.append(pd.to_datetime(s['snapshot_date']))
            values.append(float(s['portfolio_value']))
    else:
        for h in executed:
            r_d = pd.to_datetime(h['rebalance_date'])
            hld = h.get("discrete_holdings", {})
            csh = float(h.get("cash", 0.0)) if h.get("cash") is not None else 0.0
            tickers = list(hld.keys())
            md = db.get_latest_market_data(tickers, target_date=r_d.date())
            val = csh + sum(l * md.get(t, {}).get("close", 0) * md.get(t, {}).get("lot_size", 1) for t, l in hld.items() if t in md)
            dates.append(r_d)
            values.append(val)

    dates.append(pd.to_datetime(get_active_date()))
    invested = sum(l * market_data.get(t, {}).get("close", 0) * market_data.get(t, {}).get("lot_size", 1) for t, l in current_holdings.items())
    values.append(cash + invested)

    fig = go.Figure(go.Scatter(x=dates, y=values, mode="lines+markers", line=dict(width=2), marker=dict(size=6)))
    fig.add_hline(y=initial_balance, line_dash="dash", line_color="gray")
    fig.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10), xaxis_rangeslider_visible=False)
    fig.update_yaxes(tickformat=", .0f")

    fig.update_xaxes(
        range=[open_date, pd.to_datetime(get_active_date())],
        tickformat="%d\n%b",
        tick0=dates[0],
        ticklabelmode="period",
        showgrid=True,
        ticks="inside"
    )
    st.markdown("#### График стоимости портфеля")
    st.plotly_chart(fig, width='stretch')


def render_portfolio_history(db: DatabaseManager) -> None:
    """
    Отображает архив завершённых портфелей.
    
    Args:
        db: Экземпляр DatabaseManager
    """
    from .config import format_money
    from .ui_components import render_portfolio_structure_ui
    st.title("Архив портфелей")
    session_id = st.session_state.get("session_id")
    if not session_id:
        st.warning("Сессия не активна")
        return
    closed_ports = db.get_closed_portfolios(session_id)
    if not closed_ports:
        st.info("В архиве нет завершённых портфелей")
        return
    for p in closed_ports:
        created_dt = pd.to_datetime(p['created_at'])
        closed_dt = pd.to_datetime(p['closed_at']) if p['closed_at'] else None
        ref_date = closed_dt.date()
        title = f"{created_dt.strftime('%d.%m.%Y')} → {ref_date.strftime('%d.%m.%Y')}"

        with st.expander(title, expanded=False):
            snap = db.get_portfolio_snapshot(p['portfolio_id'], ref_date)
            if snap:
                current_holdings = snap.get("holdings", {})
                last_cash = float(snap["cash"])
                current_total_value = float(snap["portfolio_value"])
            else:
                current_holdings = {}
                last_cash = 0.0
                current_total_value = float(p["initial_balance"])

            tickers = list(current_holdings.keys())
            market_data = db.get_latest_market_data(tickers, target_date=ref_date) if tickers else {}
            initial_balance = float(p["initial_balance"])
            open_date = created_dt.date()
            horizon_days = int(p.get("horizon_days", 60))

            snap_hist = db.get_portfolio_history(p['portfolio_id'], open_date, ref_date)
            metrics = {}
            has_history = len(snap_hist) >= 2

            if has_history:
                values = [float(s['portfolio_value']) for s in snap_hist]
                values_series = pd.Series(values)
                returns = values_series.pct_change().dropna()
                if len(returns) >= 1:
                    total_return = values[-1] / initial_balance - 1.0
                    n_days = len(returns)
                    ann_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0
                    volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
                    sharpe = ann_return / volatility if volatility > 0 else 0.0
                    cum = (1 + returns).cumprod()
                    max_dd = (cum / cum.cummax() - 1).min()
                    var_95 = np.percentile(returns, 5)
                    tail = returns[returns <= var_95]
                    cvar_95 = tail.mean() if not tail.empty else var_95
                    downside_std = np.std(returns[returns < 0]) * np.sqrt(252) if len(returns[returns < 0]) > 1 else 0.0
                    sortino = ann_return / downside_std if downside_std > 0 else 0.0
                    metrics = {
                        'total_return': total_return, 'annual_return': ann_return, 'volatility': volatility,
                        'sharpe': sharpe, 'max_drawdown': max_dd, 'var_95': var_95, 'cvar_95': cvar_95, 'sortino': sortino
                    }

            current_total_return = (current_total_value / initial_balance - 1.0) if initial_balance > 0 else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Общий капитал (RUB)", format_money(current_total_value), delta=f"{current_total_return:.2%}")
            c2.metric("Стоимость активов (RUB)", format_money(current_total_value - last_cash))
            c3.metric("Свободный остаток (RUB)", format_money(last_cash))

            c4, c5, c6 = st.columns(3)
            c4.metric("Дата открытия", open_date.strftime("%d.%m.%Y"))
            c5.metric("Дата закрытия", ref_date.strftime("%d.%m.%Y"))
            c6.metric("Горизонт (торг. дни)", f"{horizon_days} дн.")

            st.space()
            st.header("Оценка эффективности")
            if has_history:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Доходность", f"{metrics.get('total_return', 0):.2%}")
                m2.metric("Годовая доходность", f"{metrics.get('annual_return', 0):.2%}")
                m3.metric("Волатильность (год.)", f"{metrics.get('volatility', 0):.2%}")
                m4.metric("Макс. просадка", f"{metrics.get('max_drawdown', 0):.2%}")
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Коэф. Шарпа", f"{metrics.get('sharpe', 0):.2f}")
                m6.metric("Коэф. Сортино", f"{metrics.get('sortino', 0):.2f}")
                m7.metric("VaR (95%)", f"{metrics.get('var_95', 0):.2%}")
                m8.metric("CVaR (95%)", f"{metrics.get('cvar_95', 0):.2%}")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Доходность", "–"); m2.metric("Годовая доходность", "–")
                m3.metric("Волатильность (год.)", "–"); m4.metric("Макс. просадка", "–")
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Коэф. Шарпа", "–"); m6.metric("Коэф. Сортино", "–")
                m7.metric("VaR (95%)", "–"); m8.metric("CVaR (95%)", "–")

            if has_history:
                dates, values = [], []
                for s in snap_hist:
                    dates.append(pd.to_datetime(s['snapshot_date']))
                    values.append(float(s['portfolio_value']))
                dates.append(pd.to_datetime(ref_date))
                invested = sum(l * market_data.get(t, {}).get("close", 0) * market_data.get(t, {}).get("lot_size", 1) for t, l in current_holdings.items())
                values.append(last_cash + invested)

                fig = go.Figure(go.Scatter(x=dates, y=values, mode="lines+markers", line=dict(width=2), marker=dict(size=6)))
                fig.add_hline(y=initial_balance, line_dash="dash", line_color="gray")
                fig.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10), xaxis_rangeslider_visible=False)
                fig.update_yaxes(tickformat=", .0f")
                fig.update_xaxes(range=[pd.to_datetime(open_date), pd.to_datetime(ref_date)], tickformat="%d\n%b", ticklabelmode="period", showgrid=True, ticks="inside")
                st.markdown("#### График стоимости портфеля")
                st.plotly_chart(fig, width='stretch', key=f"hist_chart_{p['portfolio_id']}")

            st.space()
            st.header("Структура портфеля")
            render_portfolio_structure_ui(current_holdings, market_data, current_total_value, key=f"hist_struct_{p['portfolio_id']}")
            st.space()

            with st.expander("История операций", expanded=False):
                history = db.get_rebalance_history_detailed(p['portfolio_id'])
                executed = [h for h in history if h.get("status") == "executed"]
                if executed:
                    chron_executed = sorted(executed, key=lambda x: x['rebalance_date'])
                    for i, rec in enumerate(chron_executed):
                        prev_holdings = chron_executed[i-1].get("discrete_holdings", {}) if i > 0 else {}
                        text = "Ребалансировка" if prev_holdings else "Формирование портфеля"
                        reason_display = f" ({rec.get('reason')})" if i > 0 and rec.get('reason') else ""
                        with st.expander(f"{rec['rebalance_date'].strftime('%d.%m.%Y')} – {text}{reason_display}", expanded=False):
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Стоимость активов (RUB)", format_money(rec.get("portfolio_value") or 0))
                            c2.metric("Свободный остаток (RUB)", format_money(rec.get("cash", 0) or 0))
                            c3.metric("Комиссия (RUB)", format_money(rec.get("commission", 0) or 0))
                            render_rebalance_operations(db, rec, prev_holdings)
                else:
                    st.info("Операции еще не проводились")

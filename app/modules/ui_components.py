"""Модуль UI компонентов для отображения портфеля и прогнозов."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bsac_portfolio.database.db_manager import DatabaseManager
from bsac_portfolio.portfolio.portfolio_logic import calculate_rebalance_actions


def render_portfolio_structure_ui(holdings: dict, market_data: dict, total_value: float = None, key: str = None) -> float:
    """
    Отображает структуру портфеля в виде круговой диаграммы и таблицы.
    
    Args:
        holdings: Словарь с количеством лотов по каждому тикеру
        market_data: Словарь с рыночными данными по каждому тикеру
        total_value: Общая стоимость портфеля
        key: Уникальный ключ для компонентов Streamlit

    Returns:
        Суммарная стоимость инвестированных средств
    """
    from .config import format_money
    rows, colors = [], []
    invested_sum = 0.0
    for t, lots in holdings.items():
        info = market_data.get(t)
        if info:
            cost = lots * info["close"] * info["lot_size"]
            invested_sum += cost
            rows.append({"Тикер": t, "Кол-во лотов": lots, "Цена": info["close"], "Стоимость": cost})
            colors.append(info.get("color", "#cccccc"))
    df = pd.DataFrame(rows)
    if df.empty:
        return
    denom = total_value if total_value and total_value > 0 else invested_sum
    df["Доля"] = df["Стоимость"] / denom if denom > 0 else 0.0
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(go.Pie(labels=df["Тикер"], values=df["Стоимость"], hole=0.4, textinfo="label+percent", textposition="outside", marker=dict(colors=colors), hovertemplate="<b>%{label}</b><br>Стоимость: %{value:,.0f} RUB<br>Доля: %{percent}<extra></extra>"))
        fig.update_layout(height=340, width=400, margin=dict(t=0, b=40, l=20, r=40), showlegend=False)
        st.plotly_chart(fig, width='stretch', key=key)
    with col2:
        df_disp = df.copy()
        df_disp = df_disp.sort_values(by='Доля', ascending=False)
        df_disp["Доля"] = df_disp["Доля"].apply(lambda x: f"{x:.2%}")
        df_disp["Стоимость"] = df_disp["Стоимость"].apply(format_money)
        df_disp["Цена"] = df_disp["Цена"].apply(lambda x: f"{x:,.2f}")
        st.dataframe(df_disp[["Тикер", "Кол-во лотов", "Цена", "Стоимость", "Доля"]], width='stretch', hide_index=True, column_config={"Кол-во лотов": st.column_config.TextColumn(alignment="right"), "Цена": st.column_config.TextColumn(alignment="right"), "Стоимость": st.column_config.TextColumn(alignment="right"), "Доля": st.column_config.TextColumn(alignment="right")})
    return invested_sum


def render_forecast_plot(mc_result: dict, cash: float, current_date: pd.Timestamp, horizon_days: int) -> None:
    """
    Отображает график прогноза стоимости портфеля с доверительным интервалом.
    
    Args:
        mc_result: Результат моделирования Монте-Карло
        cash: Текущая сумма наличных средств в портфеле
        current_date: Текущая дата
        horizon_days: Горизонт прогнозирования в днях
    """
    from .date_utils import get_trading_days_range
    if not mc_result or mc_result.get("cum_paths") is None:
        st.warning("Недостаточно данных для прогноза")
        return
    d_forecast = get_trading_days_range(current_date, horizon_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d_forecast, y=[v + cash for v in mc_result["p5"]], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d_forecast, y=[v + cash for v in mc_result["p95"]], mode="lines", name="Доверительный интервал (5%-95%)", line=dict(width=0), fill="tonexty", fillcolor="rgba(31, 119, 180, 0.18)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d_forecast, y=[v + cash for v in mc_result["p50"]], mode="lines", name="Базовый сценарий (медиана)", line=dict(color="#1f77b4", width=3.5)))
    fig.add_hline(y=mc_result["start_val"] + cash, line_dash="dot", line_color="rgba(90, 90, 90, 0.9)")
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), height=350, margin=dict(t=30, b=10, l=10, r=10))
    fig.update_yaxes(tickformat=", .0f")
    fig.update_xaxes(
            range=[current_date, d_forecast[-1]],
            tickformat="%d\n%b",
            ticklabelmode="period",
            showgrid=True,
            ticks="inside"
    )
    st.plotly_chart(fig, width='stretch')


def render_rebalance_operations(db: DatabaseManager, rec: dict, previous_holdings: dict = None) -> tuple[bool, float]:
    """
    Отображает операции ребалансировки в виде таблицы.
    
    Args:
        db: Экземпляр DatabaseManager
        rec: Результат проверки критериев ребалансировки
        previous_holdings: Предыдущая структура портфеля

    Returns:
        Кортеж (есть ли изменения в структуре портфеля, суммарная стоимость операций ребалансировки)
    """
    if previous_holdings is None:
        previous_holdings = {}
    curr_holdings = rec.get("discrete_holdings", {})
    tickers = list(set(list(curr_holdings.keys()) + list(previous_holdings.keys())))
    market_data = db.get_latest_market_data(tickers, target_date=pd.to_datetime(rec["rebalance_date"]).date())
    actions_df, turnover = calculate_rebalance_actions(curr_holdings, previous_holdings, market_data)

    if not actions_df.empty:
        if "Цена" in actions_df.columns:
            actions_df["Цена"] = actions_df["Цена"].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "0.00")
        if "Стоимость" in actions_df.columns:
            actions_df["Стоимость"] = actions_df["Стоимость"].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "0.00")
        st.dataframe(actions_df, width='stretch', hide_index=True, column_config={"Цена": st.column_config.TextColumn(alignment="right"), "Стоимость": st.column_config.TextColumn(alignment="right")})
        return True, turnover

    st.info("Нет изменений в структуре портфеля")
    return False, 0.0

"""Модуль скринера акций."""

from datetime import timedelta

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bsac_portfolio.data_preparation.feature_engineer import FeatureEngineer
from bsac_portfolio.database.db_manager import DatabaseManager
from bsac_portfolio.portfolio.portfolio_logic import calculate_asset_metrics


@st.fragment
def run_screener_calculation(db: DatabaseManager, selected_ticker: str, date_range: tuple, params: dict) -> None:
    """
    Выполняет расчёт технических индикаторов для выбранного тикера.
    
    Args:
        db: Экземпляр DatabaseManager
        selected_ticker: Выбранный тикер
        date_range: Кортеж с начальной и конечной датой
        params: Словарь с параметрами расчёта индикаторов
    """
    from .data_cache import get_market_data
    with st.spinner("Загрузка и расчёт"):
        start_str = date_range[0].strftime("%Y-%m-%d")
        end_str = date_range[1].strftime("%Y-%m-%d")

        max_warmup = 0
        if params['use_sma'] and params['sma_input']:
            try:
                periods = [int(x.strip()) for x in params['sma_input'].split(',') if x.strip().isdigit()]
                if periods:
                    max_warmup = max(max_warmup, max(periods))
            except ValueError:
                pass

        max_warmup = max(max_warmup, params['macd_slow'], params['bb_period'],
                         params['atr_period'], params['vol_period'], params['rsi_period']) + 10

        df_raw = get_market_data(db, [selected_ticker], start_str, end_str, warmup_days=max_warmup)

        if df_raw.empty:
            st.error("Нет данных в БД за выбранный период.")
            st.session_state.screener_results = None
            return

        sma_periods = []
        if params['use_sma'] and params['sma_input']:
            try:
                sma_periods = [int(x.strip()) for x in params['sma_input'].split(',') if x.strip().isdigit()]
            except ValueError:
                st.error("Неверный формат SMA.")
                return

        engineer = FeatureEngineer(df_raw)
        df_full = engineer.calculate_indicators(
            sma_periods=sma_periods if sma_periods else [10, 20],
            ema_fast=params['macd_fast'],
            ema_slow=params['macd_slow'],
            macd_signal=params['macd_signal'],
            bb_period=params['bb_period'],
            bb_std=params['bb_std'],
            atr_period=params['atr_period'],
            rsi_period=params['rsi_period'],
            volatility_period=params['vol_period'],
            drop_na=False
        )

        df_filtered = df_full[
            df_full['date'].dt.date.between(date_range[0], date_range[1])
        ].reset_index(drop=True)

        if df_filtered.empty:
            st.warning("После фильтрации по датам не осталось данных.")
            st.session_state.screener_results = None
            return

        price_data = df_filtered.set_index('date')['close']
        returns = price_data.pct_change().dropna()
        metrics = calculate_asset_metrics(price_data, returns)

        n_rows = 2
        row_heights = [0.5, 0.2]

        if params['use_rsi']:
            n_rows += 1
            row_heights.append(0.15)
        if params['use_macd']:
            n_rows += 1
            row_heights.append(0.15)
        if params['use_atr']:
            n_rows += 1
            row_heights.append(0.15)
        if params['use_vol']:
            n_rows += 1
            row_heights.append(0.15)

        total_h = sum(row_heights)
        row_heights_norm = [h/total_h for h in row_heights]

        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights_norm
        )
        fig.add_trace(go.Candlestick(
            x=df_filtered['date'],
            open=df_filtered['open'], high=df_filtered['high'],
            low=df_filtered['low'], close=df_filtered['close'],
            name='Цена'
        ), row=1, col=1)

        if params['use_sma']:
            for win in sma_periods:
                col_name = f'sma_{win}'
                if col_name in df_filtered.columns:
                    fig.add_trace(go.Scatter(
                        x=df_filtered['date'], y=df_filtered[col_name],
                        name=f'SMA {win}', line=dict(width=1.5)
                    ), row=1, col=1)

        if params['use_bb']:
            bb_color = 'rgba(255, 165, 0, 0.3)'
            bb_fill = 'rgba(255, 165, 0, 0.1)'

            if 'bb_upper' in df_filtered.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_filtered['date'],
                        y=df_filtered['bb_upper'],
                        name='BB Upper',
                        line=dict(color='orange', width=1, dash='dot'),
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
            if 'bb_lower' in df_filtered.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_filtered['date'],
                        y=df_filtered['bb_lower'],
                        name='BB Lower',
                        line=dict(color='orange', width=1, dash='dot'),
                        fill='tonexty',
                        fillcolor=bb_fill,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
             )
        colors = ['green' if c >= o else 'red' for c, o in zip(df_filtered['close'], df_filtered['open'])]
        fig.add_trace(go.Bar(
            x=df_filtered['date'], y=df_filtered['volume'],
            name='Объём', marker_color=colors, opacity=0.8, showlegend=False
        ), row=2, col=1)

        current_row = 3

        if params['use_rsi']:
            rsi_col = f'rsi_{params["rsi_period"]}'
            if rsi_col in df_filtered.columns:
                fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered[rsi_col], name='RSI', line=dict(color='purple')), row=current_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
                fig.update_yaxes(title_text="RSI", row=current_row, col=1)
                current_row += 1

        if params['use_macd']:
            if 'macd' in df_filtered.columns:
                fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['macd'], name='MACD', line=dict(color='blue')), row=current_row, col=1)
            if 'macd_signal' in df_filtered.columns:
                fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['macd_signal'], name='Signal', line=dict(color='orange')), row=current_row, col=1)
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
            current_row += 1

        if params['use_atr'] and f'atr_{params["atr_period"]}' in df_filtered.columns:
             fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered[f'atr_{params["atr_period"]}'], name='ATR', line=dict(color='darkorange')), row=current_row, col=1)
             fig.update_yaxes(title_text="ATR", row=current_row, col=1)
             current_row += 1

        if params['use_vol'] and f'volatility_{params["vol_period"]}' in df_filtered.columns:
             fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered[f'volatility_{params["vol_period"]}'], name='Volatility', line=dict(color='indianred')), row=current_row, col=1)
             fig.update_yaxes(title_text="Vol", row=current_row, col=1)
             current_row += 1

        fig.update_layout(
            height=200 * n_rows,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.session_state.screener_results = {
            'df': df_filtered,
            'metrics': metrics,
            'fig': fig,
            'params': params.copy(),
            'ticker': selected_ticker
        }


@st.fragment
def render_screener(db: DatabaseManager) -> None:
    """
    Отображает интерфейс скринера акций.
    
    Args:
        db: Экземпляр DatabaseManager
    """
    from .config import MIN_DATE
    from .data_cache import get_available_tickers, get_security_info
    from .date_utils import get_active_date
    st.title("Скринер акций")

    if 'screener_results' not in st.session_state:
        st.session_state.screener_results = None

    tickers_list = get_available_tickers(db)

    selected_ticker = st.selectbox(
        "Тикер",
        [""] + tickers_list,
        index=0,
        key="screener_ticker_select"
    )

    if not selected_ticker:
        st.info("Чтобы начать, выберите тикер акции из списка выше")
    else:
        security_info = get_security_info(db, selected_ticker)

        if security_info:
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)

            with col_info1:
                st.metric("Тикер", selected_ticker)

            with col_info2:
                st.metric("Название", security_info.get('short_name', ''))

            with col_info3:
                st.metric("Размер лота", int(security_info.get('lot_size', 1)))

            with col_info4:
                st.metric("ISIN", security_info.get('isin', 'N/A'))

            col_info5, col_info6 = st.columns(2)

            with col_info5:
                st.metric("Полное название", security_info.get('name', ''))
            with col_info6:
                st.metric("Отрасль", security_info.get('sector', ''))

        else:
            st.info("Информация об активе отсутствует в базе данных.")

        st.space()
        current_date = get_active_date()
        date_range = st.date_input(
            "Период",
            value=(current_date - timedelta(days=365), current_date),
            min_value=MIN_DATE, max_value=current_date, format="DD.MM.YYYY",
            key="screener_date_range"
        )

        st.subheader("Технические индикаторы")
        col1, col2 = st.columns(2)

        current_params = {}

        with col1:
            current_params['use_sma'] = st.toggle("SMA (Скользящие средние)", value=False, key="chk_sma")
            if current_params['use_sma']:
                current_params['sma_input'] = st.text_input("Периоды SMA (через запятую)", value="20", key="inp_sma")
            else:
                current_params['sma_input'] = ""

            current_params['use_rsi'] = st.toggle("RSI (Индекс относительной силы)", value=False, key="chk_rsi")
            if current_params['use_rsi']:
                current_params['rsi_period'] = st.number_input("Период RSI", min_value=2, max_value=100, value=14, key="num_rsi")
            else:
                current_params['rsi_period'] = 14

            current_params['use_bb'] = st.toggle("Bollinger Bands", value=False, key="chk_bb")
            if current_params['use_bb']:
                current_params['bb_period'] = st.number_input("Период BB", min_value=2, max_value=100, value=20, key="num_bb_p")
                current_params['bb_std'] = st.number_input("Отклонение BB (std)", min_value=0.5, max_value=5.0, step=0.1, value=2.0, key="num_bb_s")
            else:
                current_params['bb_period'] = 20
                current_params['bb_std'] = 2.0

        with col2:
            current_params['use_macd'] = st.toggle("MACD", value=False, key="chk_macd")
            if current_params['use_macd']:
                current_params['macd_fast'] = st.number_input("Fast EMA", min_value=2, max_value=50, value=12, key="num_macd_f")
                current_params['macd_slow'] = st.number_input("Slow EMA", min_value=2, max_value=100, value=26, key="num_macd_s")
                current_params['macd_signal'] = st.number_input("Signal SMA", min_value=2, max_value=50, value=9, key="num_macd_sig")
            else:
                current_params['macd_fast'] = 12
                current_params['macd_slow'] = 26
                current_params['macd_signal'] = 9

            current_params['use_atr'] = st.toggle("ATR (Волатильность)", value=False, key="chk_atr")
            if current_params['use_atr']:
                current_params['atr_period'] = st.number_input("Период ATR", min_value=2, max_value=100, value=14, key="num_atr")
            else:
                current_params['atr_period'] = 14

            current_params['use_vol'] = st.toggle("Volatility (Std Dev)", value=False, key="chk_vol")
            if current_params['use_vol']:
                current_params['vol_period'] = st.number_input("Период Volatility", min_value=2, max_value=100, value=20, key="num_vol")
            else:
                current_params['vol_period'] = 20

        if st.button("Продолжить", type="primary", width='stretch', key="btn_run_screener"):
            run_screener_calculation(db, selected_ticker, date_range, current_params)

        if st.session_state.screener_results:
            res = st.session_state.screener_results

            if res['metrics']:
                c1, c2, c3 = st.columns(3)
                c1.metric("Общая доходность", f"{res['metrics']['total_return']:.2%}")
                c1.metric("Макс. просадка", f"{res['metrics']['max_drawdown']:.2%}")
                c2.metric("Годовая доходность", f"{res['metrics']['annualized_return']:.2%}")
                c2.metric("VaR (95%)", f"{res['metrics']['var_95']:.2%}")
                c3.metric("Волатильность (год.)", f"{res['metrics']['annualized_volatility']:.2%}")
                c3.metric("CVaR (95%)", f"{res['metrics']['cvar_95']:.2%}")

            st.plotly_chart(res['fig'], width='stretch', config={'displayModeBar': False})

            with st.expander("Показать данные"):
                st.dataframe(res['df'], width='stretch')

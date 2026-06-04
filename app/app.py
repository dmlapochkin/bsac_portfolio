from datetime import date

import streamlit as st

from modules.config import DSN, APP_MODE, MIN_DATE, format_money
from modules.data_cache import db, init_data_cache
from modules.date_utils import get_active_date, _change_date, _reset_date
from modules.portfolio_manager import render_portfolio_tab
from modules.portfolio_dashboard import render_portfolio_history
from modules.screener import render_screener
from modules.session_manager import initialize_session, exit_session, reset_session


db, profiles_df = init_data_cache(DSN)


st.set_page_config(page_title="BSAC Portfolio Management", layout="wide", page_icon="💸", initial_sidebar_state="expanded")


session_id = initialize_session()

if 'portfolio_id' not in st.session_state or not st.session_state.get('portfolio_id'):
    if session_id:
        try:
            active_p = db.get_active_portfolio(session_id)
            if active_p:
                st.session_state.portfolio_id = active_p['portfolio_id']
            else:
                st.session_state.portfolio_id = None
        except Exception as e:
            st.session_state.portfolio_id = None
    else:
        st.session_state.portfolio_id = None

with st.sidebar:
    if APP_MODE == "demo":
        st.header("Демо-режим")
        cur_date = get_active_date()
        new_date = st.date_input("Текущая дата", value=cur_date, min_value=MIN_DATE, max_value=date.today())
        if new_date != cur_date:
            st.session_state.demo_date_override = new_date
            st.cache_data.clear()
        c_p, c_n = st.columns(2)
        with c_p:
            st.button("-1 день", use_container_width=True, on_click=lambda: _change_date(-1))
        with c_n:
            st.button("+1 день", use_container_width=True, on_click=lambda: _change_date(1))
        st.button("Сбросить дату", use_container_width=True, on_click=_reset_date)
        st.space()

    if 'risk_profile_id' in st.session_state:
        profile = db.get_risk_profile(st.session_state.risk_profile_id)
        active_p = db.get_active_portfolio(session_id)
        current_balance = 0.0
        if active_p:
            snap = db.get_latest_portfolio_state(active_p['portfolio_id'])
            current_balance = float(snap['portfolio_value']) if snap else float(db.get_session(session_id)['current_balance'])
        else:
            current_balance = float(db.get_session(session_id)['current_balance'])
        st.metric("Дата", get_active_date().strftime("%d.%m.%Y"))
        st.metric("Профиль риска", f"{profile['display_name']}")
        st.metric("Общий капитал (RUB)", f"{format_money(current_balance)}")

        with st.popover("Изменить конфигурацию", type="primary"):
            session_data = db.get_session(session_id)
            active_p_conf = db.get_active_portfolio(session_id) if session_id else None
            has_active_portfolio = active_p_conf is not None
            cfg_balance = float(session_data['current_balance'])
            new_balance = st.number_input("Общий капитал (RUB)", value=cfg_balance, step=10000.00, disabled=has_active_portfolio, help="Изменение капитала недоступно при наличии активного портфеля.")
            new_planned = st.number_input("Частота ребалансировки (дн.)", value=int(session_data["rebalance_policy_planned"]), step=1)
            new_benchmark = st.number_input("Доп. отклонение бенчмарка (%)", value=float(session_data["rebalance_policy_benchmark"]) * 100, step=1.0)
            new_weights = st.number_input("Доп. отклонение структуры (%)", value=float(session_data["rebalance_policy_weights"]) * 100, step=1.0)
            if st.button("Сохранить изменения", type="primary", width='stretch'):
                if new_balance != cfg_balance and has_active_portfolio:
                    st.error("Нельзя менять капитал при активном портфеле.")
                else:
                    db.update_session_params(session_id=session_id, balance=new_balance, rebalance_policy_planned=new_planned, rebalance_policy_benchmark=new_benchmark / 100, rebalance_policy_weights=new_weights / 100)
                    st.rerun()

    st.space()
    st.subheader("ID сессии")
    st.code(session_id, language="text")

    col_1, col_2 = st.columns(2)
    with col_1:
        st.button("Выйти", on_click=exit_session, width='stretch', type="primary")
    with col_2:
        st.button("Сбросить", on_click=reset_session, width='stretch', type="secondary")


tab1, tab2, tab3 = st.tabs(["Портфель", "Архив", "Скринер"])
with tab1:
    render_portfolio_tab(db)
with tab2:
    render_portfolio_history(db)
with tab3:
    render_screener(db)

import streamlit as st
import pandas as pd
import os

os.environ['NIXTLA_ID_AS_COL'] = '1'

if 'role' not in st.session_state:
    st.session_state.role = None
# Directory of the pages
# st.logo("atfp-logo.png")
st.set_page_config(
    # page_title="ATFP",
    layout="wide",
    initial_sidebar_state="auto"
)


st.logo("atfp-logo-rgb.png")

st.html("""
  <style>
    [alt=Logo] {
      height: 3rem;
    }
  </style>
        """)

p1 = st.Page(
    "pa/ATFP.py",
    title="Головна",
)
p2 = st.Page(
    "pa/Дані.py",
    title="Дані",
)
p3 = st.Page(
    "pa/Налаштування моделі.py",
    title="Налаштування моделі",
)

p4 = st.Page(
    "pa/Прогноз.py",
    title="Прогноз",
)
p5 = st.Page(
    "pa/Тест на аномалії.py",
    title="Аналіз на аномалії",
)
p6 = st.Page(
    "pa/ШІ помічник.py",
    title="ШІ помічник",
)


# st.title("ATFP - AI Timeseries Forecasting Platform")
#
# st.subheader(
#     "Вітаю, це сторінка проєкту, який є програмним застосунком для дослідників у сфері прогнозування часових рядів використовуючи методи машинного навчання. Для початку роботи натисніть розділ Дані, щоб програма отримала дані з якими ви хочете працювати та отримувати прогнози.")
# video_f = open("instruction_2.mp4", "rb")
# video_bytes = video_f.read()
# st.subheader(" ")
# st.divider()
# st.subheader("Відео інструкція користування застосунком")
# st.video(video_bytes)



# if st.session_state.role == "Аматор або професіонал":
pg = st.navigation({"":[p1, p2], "Для фахівців:":[p3, p4, p5], "Для всіх:":[p6]})
pg.run()

# if st.session_state.role == "Новачок":
#     pg = st.navigation([p1, p2, p6])
#     pg.run()

# if st.session_state.role is None:
    # st.title("Перед користуванням цим застосунком, оберіть хто Ви у сфері прогнозування часових рядів")
    #
    # role = st.selectbox("Роль:", ["Аматор або професіонал", "Новачок"])
    # if st.button("Підтвердити"):
    #     st.session_state.role = role
    #     st.rerun()




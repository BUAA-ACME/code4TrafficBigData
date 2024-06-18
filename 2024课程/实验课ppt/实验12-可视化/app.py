import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_absolute_error

@st.cache_resource
def get_con():
    return create_engine('postgresql+psycopg2://postgres:12344321@localhost:5432/traffic').connect()

@st.cache_data
def get_y_true():
    return pd.read_csv('true.csv').values

def handle_sql():
    name = st.session_state.name
    if not name:
        st.error("name can't be empty!", icon="🚨")
        return 
    if not uploaded_file:
        st.error("Please upload your file", icon="🚨")
        return 
    try:
        score = mean_absolute_error(y_true, re.values)
    except:
        st.error("Wrong file", icon="🚨")
        return 
    con.execute(text(f"INSERT INTO test1 (name, score, ts) VALUES ( '{st.session_state.name}', {score}, now());"))

con = get_con()
y_true = get_y_true()

st.title('交调预测练习-线上测试平台')
df = pd.read_sql(text("select name, min(score) as best_score, count(*) as count from test1 group by name order by best_score"), con,)
df = df.rename(columns={'name':'姓名', 'best_score':'最佳成绩(MAE)','count':'提交次数'})
st.dataframe(df)

uploaded_file = st.file_uploader("Upload CSV", type=".csv")
if uploaded_file:
    re = pd.read_csv(uploaded_file)
    st.markdown("### Data preview")
    st.dataframe(re)
    if re.shape != y_true.shape:
        st.warning(f"上传结果格式验证有误，希望的shape为{y_true.shape}，但是上传的shape为{re.shape}", icon="🚨")
with st.form(key='my_form'):
    st.text_input('Enter your name', key='name')
    submit = st.form_submit_button(label='Submit', on_click=handle_sql)
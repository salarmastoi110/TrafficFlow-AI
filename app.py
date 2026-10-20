"""TrafficFlow-AI: traffic analytics and prediction dashboard.  Run: streamlit run app.py"""
import plotly.express as px
import streamlit as st

from src.prediction import forecast_day, load_metrics, model_exists, predict_traffic
from src.preprocessing import DAYS, JUNCTIONS, WEATHER, load_data
from src.train_model import train

COLORS = {"Low": "#2e9e5b", "Moderate": "#e0b020", "High": "#e8742c", "Severe": "#d33a3a"}
ORDER = ["Low", "Moderate", "High", "Severe"]

st.set_page_config(page_title="TrafficFlow-AI", page_icon="🚦", layout="wide")


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def ensure_model():
    if not model_exists():
        train()
    return True


ensure_model()
df = get_data()

st.sidebar.title("🚦 TrafficFlow-AI")
page = st.sidebar.radio("Navigate", ["Dashboard", "Prediction", "Analytics"])
st.sidebar.caption("Hourly traffic analytics and ML-based volume forecasting.")


def dashboard():
    st.header("Traffic Dashboard")
    picks = st.multiselect("Junctions", list(JUNCTIONS), default=list(JUNCTIONS))
    d = df[df["junction"].isin(picks)]
    if d.empty:
        st.info("Select at least one junction.")
        return
    hourly = d.groupby("hour")["vehicles"].mean()
    c = st.columns(4)
    c[0].metric("Avg vehicles / hour", f"{d['vehicles'].mean():.0f}")
    c[1].metric("Peak hour", f"{int(hourly.idxmax()):02d}:00")
    c[2].metric("Busiest junction", d.groupby("junction")["vehicles"].mean().idxmax())
    c[3].metric("High/Severe share", f"{d['congestion'].isin(['High', 'Severe']).mean():.0%}")

    by_hj = d.groupby(["hour", "junction"], as_index=False)["vehicles"].mean()
    st.plotly_chart(px.line(by_hj, x="hour", y="vehicles", color="junction", title="Average hourly traffic by junction"),
                    width="stretch")
    heat = d.pivot_table(index="day_of_week", columns="hour", values="vehicles", aggfunc="mean")
    heat.index = [DAYS[i] for i in heat.index]
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                              labels=dict(x="Hour", y="Day", color="Vehicles"), title="Weekly traffic heatmap"),
                    width="stretch")


def prediction():
    st.header("Traffic Prediction")
    c1, c2, c3 = st.columns(3)
    junction = c1.selectbox("Junction", list(JUNCTIONS))
    day = c1.selectbox("Day", DAYS, index=0)
    hour = c2.slider("Hour", 0, 23, 8)
    weather = c2.selectbox("Weather", WEATHER)
    temp = c3.slider("Temperature (°C)", 5, 45, 28)
    holiday = c3.checkbox("Public holiday")
    dow = DAYS.index(day)

    res = predict_traffic(junction, hour, dow, weather, temp, holiday)
    m = st.columns(2)
    m[0].metric("Predicted vehicles / hour", res["vehicles"])
    m[1].markdown(f"**Congestion level**<h2 style='color:{COLORS[res['congestion']]};margin:0'>{res['congestion']}</h2>",
                  unsafe_allow_html=True)

    fc = forecast_day(junction, dow, weather, temp, holiday)
    fig = px.bar(fc, x="hour", y="vehicles", color="congestion", color_discrete_map=COLORS,
                 category_orders={"congestion": ORDER}, title=f"24-hour forecast: {junction}, {day}")
    fig.add_vline(x=hour, line_dash="dash")
    st.plotly_chart(fig, width="stretch")


def analytics():
    st.header("Analytics")
    mt = load_metrics()
    if mt:
        c = st.columns(3)
        c[0].metric("MAE (vehicles)", mt["mae"])
        c[1].metric("RMSE", mt["rmse"])
        c[2].metric("R² score", mt["r2"])
    a, b = st.columns(2)
    w = df.groupby("weather", as_index=False)["vehicles"].mean()
    a.plotly_chart(px.bar(w, x="weather", y="vehicles", title="Avg traffic by weather"), width="stretch")
    dist = df["congestion"].value_counts().reindex(ORDER).reset_index()
    dist.columns = ["congestion", "count"]
    b.plotly_chart(px.pie(dist, names="congestion", values="count", color="congestion", color_discrete_map=COLORS,
                          title="Congestion distribution"), width="stretch")
    daily = df.groupby([df["timestamp"].dt.date, "junction"], as_index=False)["vehicles"].sum()
    st.plotly_chart(px.line(daily, x="timestamp", y="vehicles", color="junction", title="Daily total traffic"),
                    width="stretch")
    with st.expander("Raw data"):
        st.dataframe(df.head(500), width="stretch")
        st.download_button("Download CSV", df.to_csv(index=False), "traffic_data.csv", "text/csv")


{"Dashboard": dashboard, "Prediction": prediction, "Analytics": analytics}[page]()

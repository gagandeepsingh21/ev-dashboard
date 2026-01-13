import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------
# 1. Configuration & Styling
# -----------------------------
st.set_page_config(
    page_title="EV Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a polished look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div.stButton > button:first-child { background-color: #007bff; color: white; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# 2. Data & Model Loaders
# -----------------------------
@st.cache_resource
def load_assets():
    try:
        range_model = joblib.load("ev_range_model.pkl")
        range_features = joblib.load("range_features.pkl")
        growth_model = joblib.load("ev_growth_model.pkl")
        df_range = pd.read_csv("cleaned_data_electric_range.csv")
        df_growth = pd.read_csv("cleaned_data_growth.csv")
        return range_model, range_features, growth_model, df_range, df_growth
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None, None

range_model, range_features, growth_model, df_range, df_growth = load_assets()

# -----------------------------
# 3. Sidebar Navigation
# -----------------------------
st.sidebar.title("⚡ Navigation")
app_mode = st.sidebar.radio("Choose a Tool:", ["Overview", "Range Predictor", "Growth Forecast"])

if range_model is None:
    st.warning("Please ensure model and data files are in the directory.")
    st.stop()

# -----------------------------
# 4. Main App Logic
# -----------------------------

if app_mode == "Overview":
    st.title("🚗 EV Market Insights Overview")
    
    # Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total EVs Tracked", f"{len(df_growth):,}")
    with m2:
        st.metric("Avg. Electric Range", f"{int(df_range['Electric Range'].mean())} mi")
    with m3:
        st.metric("Counties Covered", df_growth['County'].nunique())
    with m4:
        st.metric("Model Years", f"{int(df_growth['Model Year'].min())} - {int(df_growth['Model Year'].max())}")

    st.markdown("---")
    
    # Quick Distribution Chart
    st.subheader("Top EV Makes in Dataset")
    make_counts = df_growth['Make'].value_counts().head(10)
    fig_make = px.bar(make_counts, x=make_counts.index, y=make_counts.values, 
                     labels={'x': 'Manufacturer', 'y': 'Count'},
                     color=make_counts.values, color_continuous_scale='Blues')
    st.plotly_chart(fig_make, use_container_width=True)

elif app_mode == "Range Predictor":
    st.title("🔋 Electric Range Predictor")
    st.markdown("Enter vehicle specifications to estimate real-world driving range.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        # Identify base categories from one-hot encoded features
        # Assuming features are named like 'Make_TESLA', 'County_KING', etc.
        makes = sorted([f.replace('Make_', '') for f in range_features if f.startswith('Make_')])
        counties = sorted([f.replace('County_', '') for f in range_features if f.startswith('County_')])
        
        with col1:
            st.subheader("Vehicle Identity")
            in_make = st.selectbox("Manufacturer", makes if makes else ["N/A"])
            in_year = st.number_input("Model Year", min_value=1990, max_value=2035, value=2022)

        with col2:
            st.subheader("Location & Specs")
            in_county = st.selectbox("County", counties if counties else ["N/A"])
            
        submit = st.form_submit_button("Generate Prediction", use_container_width=True)

    if submit:
        # Construct the input vector matching range_features order
        input_data = {}
        for feat in range_features:
            if feat == 'Model Year': input_data[feat] = in_year

            elif feat == f"Make_{in_make}": input_data[feat] = 1
            elif feat == f"County_{in_county}": input_data[feat] = 1
            else:
                input_data[feat] = 0 # All other one-hot columns are 0
        
        input_df = pd.DataFrame([input_data])[range_features]
        prediction = range_model.predict(input_df)[0]

        # Display Result with a gauge chart
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.success(f"### Predicted ELectric Range: \n# {prediction:.1f} Miles")

        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Electric Range Capacity (Miles)"},
                gauge = {
                    'axis': {'range': [0, 500]},
                    'bar': {'color': "#00cc96"},
                    'steps': [
                        {'range': [0, 150], 'color': "#ffefef"},
                        {'range': [150, 300], 'color': "#e8f5e9"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 400}
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

elif app_mode == "Growth Forecast":
    st.title("📈 Regional Adoption Forecast")
    
    # Prepare data
    df_growth_county = df_growth.groupby(['Model Year', 'County']).size().reset_index(name='EV_Count')
    county_list = sorted(df_growth_county["County"].unique())
    
    c1, c2 = st.columns(2)
    with c1:
        sel_county = st.selectbox("Target County", county_list)
    with c2:
        sel_year = st.slider("Forecast Target Year", 2024, 2040, 2030)

    if st.button("Run Forecast Analysis", type="primary"):
        # Mapping for the model
        if 'County_Cat' not in df_growth_county.columns:
            df_growth_county['County_Cat'] = df_growth_county['County'].astype('category').cat.codes
        
        county_map = dict(zip(df_growth_county['County'], df_growth_county['County_Cat']))
        
        pred_input = pd.DataFrame({"Model Year": [sel_year], "County_Cat": [county_map[sel_county]]})
        prediction = growth_model.predict(pred_input)[0]
        
        # Historical Comparison
        hist_data = df_growth_county[df_growth_county["County"] == sel_county]
        latest_count = hist_data.iloc[-1]["EV_Count"]
        latest_year = hist_data.iloc[-1]["Model Year"]
        
        # UI Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(f"Current ({latest_year})", f"{int(latest_count):,}")
        m_col2.metric(f"Forecast ({sel_year})", f"{int(prediction):,}")
        m_col3.metric("Estimated Growth", f"{((prediction-latest_count)/latest_count*100):.1f}%", delta_color="normal")

        # Forecast Chart
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=hist_data["Model Year"], y=hist_data["EV_Count"], 
                                         name="Historical Data", line=dict(color='blue', width=3)))
        
        # Add prediction line
        future_years = [latest_year, sel_year]
        future_counts = [latest_count, prediction]
        fig_forecast.add_trace(go.Scatter(x=future_years, y=future_counts, 
                                         name="Forecast Trend", line=dict(color='red', dash='dash')))
        
        fig_forecast.update_layout(title=f"EV Adoption Trend: {sel_county}", 
                                  xaxis_title="Model Year", yaxis_title="Total Registered EVs")
        st.plotly_chart(fig_forecast, use_container_width=True)

# -----------------------------
# 5. Footer
# -----------------------------
st.markdown("---")
st.caption(f"Generated on {datetime.now().strftime('%Y-%m-%d')} | Data Source: WA State Open Data Portal")
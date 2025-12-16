import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SA FinTech Econometric Analysis",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3.5rem;
    background: linear-gradient(90deg, #0F766E, #14B8A6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1.4rem;
    color: #4B5563;
    margin-bottom: 2rem;
}
.investment-card {
    background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(15, 118, 110, 0.3);
}
.risk-card {
    background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(220, 38, 38, 0.3);
}
.growth-card {
    background: linear-gradient(135deg, #059669 0%, #10B981 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(5, 150, 105, 0.3);
}
.metric-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    margin: 0.2rem;
}
.insight-highlight {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    border-left: 5px solid #F59E0B;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.correlation-matrix {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"> SA FinTech Econometric Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Macro-Level Sector Analysis for Venture Capital & Strategic Investment</p>', unsafe_allow_html=True)

# Generate synthetic econometric data
def generate_econometric_data():
    # Market fundamentals over time
    years = [2022, 2023, 2024, 2025, 2026]
    market_fundamentals = pd.DataFrame({
        'Year': years,
        'Total_Addressable_Market_ZAR_BN': [2.10, 2.45, 2.80, 3.22, 3.70],
        'Sector_Revenue_ZAR_BN': [0.55, 0.80, 1.16, 1.68, 2.44],
        'YoY_Growth_Rate': [0.38, 0.45, 0.45, 0.45, 0.45],
        'SMME_Digital_Adoption_Rate': [0.12, 0.15, 0.18, 0.22, 0.26]
    })
    
    # Funding flow trends
    quarters = [f"{y}-Q{q}" for y in [2022, 2023, 2024] for q in [1, 2, 3, 4]]
    funding_data = pd.DataFrame({
        'Quarter': quarters[:10],  # 2022-Q1 to 2024-Q2
        'Total_Funding_Raised_ZAR_M': [45, 52, 61, 78, 95, 110, 125, 150, 180, 220],
        'Number_of_Rounds': [4, 5, 6, 7, 8, 9, 10, 12, 14, 16],
        'Avg_Round_Size_ZAR_M': [11.3, 10.4, 10.2, 11.1, 11.9, 12.2, 12.5, 12.5, 12.9, 13.8],
        'Top_Sector': ['Payments Infrastructure', 'Business Banking', 'Invoicing & Expense', 
                      'Invoicing & Expense', 'Lending & Credit', 'Lending & Credit',
                      'Cashflow Management', 'Cashflow Management', 'Embedded Finance', 'Embedded Finance']
    })
    
    # Valuation benchmarks
    valuation_data = pd.DataFrame({
        'Company': ['Invoicely', 'ZazuPay', 'SA-Books', 'QuickStoki', 'CapitFlow'],
        'ARR_ZAR_M': [10.0, 6.8, 6.8, 5.6, 5.6],
        'Last_Valuation_ZAR_M': [95.0, 45.0, 35.0, 25.0, 120.0],
        'Growth_Rate': [0.66, 0.54, 0.13, 0.40, 1.33],
        'Gross_Margin': [0.72, 0.75, 0.70, 0.68, 0.78]
    })
    
    # Calculate multiples
    valuation_data['Revenue_Multiple'] = valuation_data['Last_Valuation_ZAR_M'] / valuation_data['ARR_ZAR_M']
    valuation_data['ARR_Multiple'] = valuation_data['Revenue_Multiple'] * 0.8  # Assuming 80% recurring revenue
    
    # Sector averages
    sector_avg = pd.DataFrame({
        'Metric': ['ARR', 'Valuation', 'Revenue Multiple', 'ARR Multiple', 'Growth Rate', 'Gross Margin'],
        'Value': [6.96, 64.0, 9.4, 7.5, 0.614, 0.726]
    })
    
    # Regulatory risk scorecard
    regulatory_risk = pd.DataFrame({
        'Risk_Factor': ['POPI Act Compliance', 'SARB Licensing', 'FAIS Act Applicability', 
                       'BEE Compliance Cost', 'National Credit Act (NCA)'],
        'Weight': [0.25, 0.30, 0.20, 0.15, 0.10],
        'Score': [7, 4, 6, 8, 3],
        'Description': [
            'Data privacy enforcement increasing, clear guidance',
            'Payment system licensing complex and time-consuming',
            'Impacts companies offering financial advice',
            'Significant cost for corporate BEE rating',
            'Major barrier for fintech moving into lending'
        ]
    })
    
    regulatory_risk['Weighted_Score'] = regulatory_risk['Weight'] * regulatory_risk['Score']
    total_risk_score = regulatory_risk['Weighted_Score'].sum()
    
    # Macroeconomic correlations
    macro_correlations = pd.DataFrame({
        'Year_Quarter': ['2022-Q1', '2022-Q2', '2022-Q3', '2022-Q4', 
                        '2023-Q1', '2023-Q2', '2023-Q3', '2023-Q4',
                        '2024-Q1', '2024-Q2'],
        'FinTech_Funding_ZAR_M': [45, 52, 61, 78, 95, 110, 125, 150, 180, 220],
        'SA_GDP_Growth': [0.018, 0.005, 0.012, 0.021, 0.003, -0.007, 0.009, 0.015, 0.008, 0.011],
        'JSE_All_Share_Index': [74120, 72550, 73800, 76100, 75250, 73900, 76500, 79200, 81100, 82500],
        'Internet_Penetration': [0.72, 0.73, 0.74, 0.75, 0.77, 0.78, 0.80, 0.81, 0.83, 0.84],
        'Formal_SMME_Growth': [0.012, 0.008, 0.010, 0.015, 0.005, -0.002, 0.007, 0.013, 0.009, 0.011]
    })
    
    # Investment opportunity matrix
    opportunity_matrix = pd.DataFrame({
        'Sector': ['Payments Infrastructure', 'Business Banking', 'Invoicing & Expense',
                  'Cashflow Management', 'Lending & Credit', 'Embedded Finance'],
        'Market_Size_Score': [8, 7, 6, 5, 9, 4],  # 1-10 scale
        'Growth_Potential': [7, 8, 9, 9, 8, 10],   # 1-10 scale
        'Competitive_Intensity': [9, 8, 7, 5, 6, 4],  # Higher = more competitive
        'Regulatory_Barrier': [8, 9, 5, 6, 10, 6],    # Higher = more barriers
        'Current_Funding_Trend': [45, 52, 139, 275, 205, 400]  # Cumulative funding in R millions
    })
    
    # Calculate opportunity score
    opportunity_matrix['Opportunity_Score'] = (
        opportunity_matrix['Market_Size_Score'] * 0.25 +
        opportunity_matrix['Growth_Potential'] * 0.35 -
        opportunity_matrix['Competitive_Intensity'] * 0.20 -
        opportunity_matrix['Regulatory_Barrier'] * 0.20
    )
    
    return (market_fundamentals, funding_data, valuation_data, sector_avg, 
            regulatory_risk, total_risk_score, macro_correlations, opportunity_matrix)

# Load data
(market_fundamentals, funding_data, valuation_data, sector_avg, 
 regulatory_risk, total_risk_score, macro_correlations, opportunity_matrix) = generate_econometric_data()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=80)
    st.title("VC Investment Console")
    st.markdown("---")
    
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Sector Overview", "Valuation Analysis", "Risk Assessment", 
         "Macro Correlations", "Opportunity Matrix", "Investment Thesis"]
    )
    
    st.markdown("---")
    st.markdown("###  Key Sector Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("TAM 2024", "R2.8 BN")
    with col2:
        st.metric("Growth Rate", "45%")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Avg Multiple", "9.4x")
    with col4:
        st.metric("Risk Score", f"{total_risk_score:.1f}/10")
    
    st.markdown("---")
    st.markdown("### 🎯 Investment Horizon")
    investment_horizon = st.slider("Years", 3, 7, 5)
    min_irr = st.slider("Target IRR (%)", 25, 50, 35)

# Main content based on analysis mode
if analysis_mode == "Sector Overview":
    st.header(" Sector Health & Growth Trajectory")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="investment-card">', unsafe_allow_html=True)
        st.metric(
            "Total Addressable Market",
            f"R{market_fundamentals.iloc[2]['Total_Addressable_Market_ZAR_BN']:.1f} BN",
            f"+{((market_fundamentals.iloc[2]['Total_Addressable_Market_ZAR_BN'] - market_fundamentals.iloc[0]['Total_Addressable_Market_ZAR_BN'])/market_fundamentals.iloc[0]['Total_Addressable_Market_ZAR_BN']*100):.0f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="growth-card">', unsafe_allow_html=True)
        st.metric(
            "Sector Revenue",
            f"R{market_fundamentals.iloc[2]['Sector_Revenue_ZAR_BN']:.2f} BN",
            f"{market_fundamentals.iloc[2]['YoY_Growth_Rate']*100:.0f}% YoY"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="growth-card">', unsafe_allow_html=True)
        st.metric(
            "Digital Adoption",
            f"{market_fundamentals.iloc[2]['SMME_Digital_Adoption_Rate']*100:.0f}%",
            f"+{((market_fundamentals.iloc[2]['SMME_Digital_Adoption_Rate'] - market_fundamentals.iloc[0]['SMME_Digital_Adoption_Rate'])/market_fundamentals.iloc[0]['SMME_Digital_Adoption_Rate']*100):.0f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="risk-card">', unsafe_allow_html=True)
        st.metric(
            "Regulatory Risk",
            f"{total_risk_score:.1f}/10",
            "Moderate-High",
            delta_color="inverse"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Growth trajectory
    st.subheader(" Market Growth Trajectory")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=market_fundamentals['Year'], 
                  y=market_fundamentals['Total_Addressable_Market_ZAR_BN'],
                  name="TAM (R BN)", mode='lines+markers',
                  line=dict(color='#0F766E', width=4)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(x=market_fundamentals['Year'], 
              y=market_fundamentals['Sector_Revenue_ZAR_BN'],
              name="Sector Revenue (R BN)",
              marker_color='#14B8A6'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=market_fundamentals['Year'][1:], 
                  y=market_fundamentals['YoY_Growth_Rate'][1:] * 100,
                  name="YoY Growth %", mode='lines+markers',
                  line=dict(color='#F59E0B', width=3, dash='dash')),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Market Size & Growth Projection (2022-2026)",
        xaxis_title="Year",
        hovermode="x unified",
        height=500
    )
    
    fig.update_yaxes(title_text="Value (R Billions)", secondary_y=False)
    fig.update_yaxes(title_text="Growth Rate (%)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Funding momentum
    st.subheader(" Funding Momentum Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            funding_data,
            x='Quarter',
            y='Total_Funding_Raised_ZAR_M',
            color='Avg_Round_Size_ZAR_M',
            title='Quarterly Funding Volume',
            color_continuous_scale='Teal',
            text='Total_Funding_Raised_ZAR_M'
        )
        fig.update_traces(texttemplate='R%{text}M', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=funding_data['Quarter'],
            y=funding_data['Total_Funding_Raised_ZAR_M'].cumsum(),
            name="Cumulative Funding",
            line=dict(color='#0F766E', width=4)
        ))
        
        fig.add_trace(go.Bar(
            x=funding_data['Quarter'],
            y=funding_data['Number_of_Rounds'],
            name="# of Rounds",
            marker_color='#14B8A6',
            opacity=0.6,
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Cumulative Funding & Deal Count",
            xaxis_title="Quarter",
            yaxis_title="Cumulative Funding (R Millions)",
            yaxis2=dict(title="# of Rounds", overlaying='y', side='right'),
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif analysis_mode == "Valuation Analysis":
    st.header(" Valuation Benchmarks & Multiples")
    
    # Valuation scatter plot
    fig = px.scatter(
        valuation_data,
        x='Growth_Rate',
        y='Revenue_Multiple',
        size='Last_Valuation_ZAR_M',
        color='Company',
        hover_data=['ARR_ZAR_M', 'Gross_Margin'],
        title='Valuation Multiples vs Growth Rate (Bubble size = Valuation)',
        size_max=60
    )
    
    # Add sector average lines
    sector_avg_growth = sector_avg[sector_avg['Metric'] == 'Growth Rate']['Value'].values[0]
    sector_avg_multiple = sector_avg[sector_avg['Metric'] == 'Revenue Multiple']['Value'].values[0]
    
    fig.add_hline(y=sector_avg_multiple, line_dash="dash", line_color="gray", 
                 annotation_text=f"Sector Avg: {sector_avg_multiple:.1f}x")
    fig.add_vline(x=sector_avg_growth, line_dash="dash", line_color="gray",
                 annotation_text=f"Avg Growth: {sector_avg_growth*100:.0f}%")
    
    # Add regression line
    z = np.polyfit(valuation_data['Growth_Rate'], valuation_data['Revenue_Multiple'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=np.linspace(valuation_data['Growth_Rate'].min(), valuation_data['Growth_Rate'].max(), 10),
        y=p(np.linspace(valuation_data['Growth_Rate'].min(), valuation_data['Growth_Rate'].max(), 10)),
        mode='lines',
        name='Valuation Trend',
        line=dict(color='red', dash='dot')
    ))
    
    fig.update_layout(
        xaxis_title="Growth Rate (%)",
        yaxis_title="Revenue Multiple (x)",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed valuation table
    st.subheader(" Detailed Valuation Metrics")
    
    # Add calculated fields
    valuation_display = valuation_data.copy()
    valuation_display['Growth_Rate_%'] = (valuation_display['Growth_Rate'] * 100).round(1)
    valuation_display['Gross_Margin_%'] = (valuation_display['Gross_Margin'] * 100).round(1)
    valuation_display['Valuation/ARR_Ratio'] = valuation_display['Revenue_Multiple'].round(1)
    
    # Calculate efficiency metrics
    valuation_display['Efficiency_Score'] = (
        valuation_display['Growth_Rate'] * 0.4 +
        valuation_display['Gross_Margin'] * 0.3 +
        (1 / valuation_display['Revenue_Multiple']) * 0.3  # Inverse of multiple for efficiency
    ).round(2)
    
    # Display metrics in columns
    cols = st.columns(len(valuation_display))
    for idx, (col, (_, row)) in enumerate(zip(cols, valuation_display.iterrows())):
        with col:
            st.markdown(f'<div style="text-align: center; padding: 10px; border-radius: 10px; background: linear-gradient(135deg, #F3F4F6, #E5E7EB);">', unsafe_allow_html=True)
            st.markdown(f'**{row["Company"]}**')
            st.markdown(f'<span style="font-size: 1.5rem; font-weight: bold; color: #0F766E;">R{row["Last_Valuation_ZAR_M"]:.0f}M</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="color: #059669;">{row["Valuation/ARR_Ratio"]}x</span>', unsafe_allow_html=True)
            st.markdown(f'<small>Growth: {row["Growth_Rate_%"]}%<br>Margin: {row["Gross_Margin_%"]}%</small>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Investment implication
    st.markdown('<div class="insight-highlight">', unsafe_allow_html=True)
    st.markdown("###  Investment Implication")
    st.markdown(f"""
    **Sector Average Multiple: {sector_avg_multiple:.1f}x Revenue** | **{sector_avg[sector_avg['Metric'] == 'ARR Multiple']['Value'].values[0]:.1f}x ARR**
    
    - **Premium for Growth:** CapitFlow trades at 21.4x despite R5.6M ARR due to 133% growth
    - **Efficiency Discount:** QuickStoki at 4.5x reflects profitable but slower growth model
    - **Target Entry Point:** New investments should aim for 8-12x for growth-stage companies
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif analysis_mode == "Risk Assessment":
    st.header(" Comprehensive Risk Analysis")
    
    # Risk scorecard
    st.subheader(" Regulatory Risk Scorecard")
    
    fig = go.Figure(data=[
        go.Bar(
            x=regulatory_risk['Risk_Factor'],
            y=regulatory_risk['Weighted_Score'],
            text=regulatory_risk['Score'],
            marker_color=['#DC2626', '#EF4444', '#F59E0B', '#10B981', '#3B82F6'],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f'Regulatory Risk Components (Total Score: {total_risk_score:.2f}/10)',
        xaxis_title="Risk Factor",
        yaxis_title="Weighted Score",
        yaxis_range=[0, 3],
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk details
    for _, risk in regulatory_risk.iterrows():
        with st.expander(f"{risk['Risk_Factor']} (Score: {risk['Score']}/10 | Weight: {risk['Weight']*100}%)"):
            st.write(risk['Description'])
            st.progress(risk['Score']/10)
    
    # Risk vs reward matrix
    st.subheader(" Risk-Reward Matrix")
    
    # Create synthetic risk-reward data
    risk_reward_data = pd.DataFrame({
        'Company': valuation_data['Company'].tolist() + ['Sector Avg'],
        'Return_Potential': valuation_data['Growth_Rate'].tolist() + [sector_avg_growth],
        'Risk_Score': [6.5, 5.0, 4.0, 3.0, 8.0, 5.3],  # Synthetic risk scores
        'Valuation': valuation_data['Last_Valuation_ZAR_M'].tolist() + [sector_avg[sector_avg['Metric'] == 'Valuation']['Value'].values[0]]
    })
    
    fig = px.scatter(
        risk_reward_data,
        x='Risk_Score',
        y='Return_Potential',
        size='Valuation',
        color='Company',
        title='Risk-Return Profile (Bubble size = Valuation)',
        hover_data=['Valuation'],
        size_max=50
    )
    
    # Add quadrants
    fig.add_hline(y=risk_reward_data['Return_Potential'].mean(), line_dash="dash", line_color="gray")
    fig.add_vline(x=risk_reward_data['Risk_Score'].mean(), line_dash="dash", line_color="gray")
    
    fig.update_layout(
        xaxis_title="Risk Score (Higher = Riskier)",
        yaxis_title="Return Potential (Growth Rate)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mitigation strategies
    st.subheader(" Risk Mitigation Strategies")
    
    mitigation_strategies = pd.DataFrame({
        'Risk': ['SARB Licensing', 'NCA Compliance', 'BEE Requirements', 'POPI Compliance'],
        'Mitigation': [
            'Partner with licensed entities initially',
            'Start with advisory services, add lending later',
            'Focus on Level 4-6 targets initially',
            'Build compliance into product from day one'
        ],
        'Cost_Impact': ['Medium', 'High', 'Medium', 'Low'],
        'Timeline': ['6-12 months', '12-18 months', '3-6 months', '3 months']
    })
    
    for _, strategy in mitigation_strategies.iterrows():
        col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
        with col1:
            st.markdown(f"**{strategy['Risk']}**")
        with col2:
            st.markdown(strategy['Mitigation'])
        with col3:
            st.markdown(f"`{strategy['Cost_Impact']}`")
        with col4:
            st.markdown(f"`{strategy['Timeline']}`")
        st.divider()

elif analysis_mode == "Macro Correlations":
    st.header(" Macroeconomic Correlations")
    
    # Correlation matrix
    st.subheader(" Correlation Matrix")
    
    correlation_data = macro_correlations[['FinTech_Funding_ZAR_M', 'SA_GDP_Growth', 
                                         'JSE_All_Share_Index', 'Internet_Penetration',
                                         'Formal_SMME_Growth']].copy()
    correlation_data.columns = ['FinTech Funding', 'GDP Growth', 'JSE Index', 
                               'Internet Penetration', 'SMME Growth']
    
    corr_matrix = correlation_data.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu',
        aspect="auto",
        title='Correlation Matrix: FinTech Funding vs Macro Indicators'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key correlation insights
    st.markdown('<div class="insight-highlight">', unsafe_allow_html=True)
    st.markdown("### 🔍 Key Correlation Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "FinTech vs SMME Growth",
            f"{corr_matrix.loc['FinTech Funding', 'SMME Growth']:.2f}",
            "Strong Positive"
        )
    
    with col2:
        st.metric(
            "FinTech vs GDP Growth",
            f"{corr_matrix.loc['FinTech Funding', 'GDP Growth']:.2f}",
            "Moderate Positive"
        )
    
    with col3:
        st.metric(
            "FinTech vs Internet Penetration",
            f"{corr_matrix.loc['FinTech Funding', 'Internet Penetration']:.2f}",
            "Strong Positive"
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Time series correlations
    st.subheader(" Time Series Analysis")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=('FinTech Funding vs Economic Indicators', 
                                      'Growth Rate Comparisons'))
    
    # Top plot: Funding vs indicators
    fig.add_trace(
        go.Scatter(x=macro_correlations['Year_Quarter'], 
                  y=macro_correlations['FinTech_Funding_ZAR_M'],
                  name="FinTech Funding (R M)",
                  line=dict(color='#0F766E', width=4)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=macro_correlations['Year_Quarter'], 
                  y=macro_correlations['SA_GDP_Growth'] * 1000,  # Scale for visibility
                  name="GDP Growth (scaled)",
                  line=dict(color='#EF4444', width=2, dash='dot'),
                  yaxis='y2'),
        row=1, col=1
    )
    
    # Bottom plot: Growth rates
    fig.add_trace(
        go.Scatter(x=macro_correlations['Year_Quarter'], 
                  y=macro_correlations['Formal_SMME_Growth'] * 100,
                  name="SMME Growth (%)",
                  line=dict(color='#10B981', width=3)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=macro_correlations['Year_Quarter'], 
              y=macro_correlations['Internet_Penetration'] * 100,
              name="Internet Penetration (%)",
              marker_color='#3B82F6',
              opacity=0.6),
        row=2, col=1
    )
    
    fig.update_layout(height=700, showlegend=True)
    fig.update_yaxes(title_text="Funding (R Millions)", row=1, col=1)
    fig.update_yaxes(title_text="GDP Growth (scaled)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Growth Rate (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

elif analysis_mode == "Opportunity Matrix":
    st.header(" Investment Opportunity Matrix")
    
    # Bubble chart of opportunities
    fig = px.scatter(
        opportunity_matrix,
        x='Market_Size_Score',
        y='Growth_Potential',
        size='Current_Funding_Trend',
        color='Opportunity_Score',
        hover_name='Sector',
        hover_data=['Competitive_Intensity', 'Regulatory_Barrier'],
        title='FinTech Sector Opportunity Matrix',
        color_continuous_scale='Viridis',
        size_max=40
    )
    
    # Add quadrant lines
    fig.add_hline(y=opportunity_matrix['Growth_Potential'].mean(), 
                 line_dash="dash", line_color="gray")
    fig.add_vline(x=opportunity_matrix['Market_Size_Score'].mean(), 
                 line_dash="dash", line_color="gray")
    
    # Add quadrant labels
    fig.add_annotation(x=4, y=9, text="High Growth / Small Market", 
                      showarrow=False, font=dict(size=10, color="gray"))
    fig.add_annotation(x=4, y=5, text="Low Growth / Small Market", 
                      showarrow=False, font=dict(size=10, color="gray"))
    fig.add_annotation(x=8, y=9, text="High Growth / Large Market", 
                      showarrow=False, font=dict(size=10, color="green"))
    fig.add_annotation(x=8, y=5, text="Low Growth / Large Market", 
                      showarrow=False, font=dict(size=10, color="gray"))
    
    fig.update_layout(
        xaxis_title="Market Size Score (1-10)",
        yaxis_title="Growth Potential Score (1-10)",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed opportunity analysis
    st.subheader(" Opportunity Deep Dive")
    
    # Sort by opportunity score
    opportunity_sorted = opportunity_matrix.sort_values('Opportunity_Score', ascending=False)
    
    for idx, (_, opp) in enumerate(opportunity_sorted.iterrows(), 1):
        with st.expander(f"{idx}. {opp['Sector']} (Opportunity Score: {opp['Opportunity_Score']:.1f})"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Market Size", f"{opp['Market_Size_Score']}/10")
                st.progress(opp['Market_Size_Score']/10)
            
            with col2:
                st.metric("Growth Potential", f"{opp['Growth_Potential']}/10")
                st.progress(opp['Growth_Potential']/10)
            
            with col3:
                st.metric("Competition", f"{opp['Competitive_Intensity']}/10", 
                         delta_color="inverse")
                st.progress(opp['Competitive_Intensity']/10)
            
            with col4:
                st.metric("Funding Trend", f"R{opp['Current_Funding_Trend']}M")
            
            # Investment recommendation
            if opp['Opportunity_Score'] > 6:
                st.success(f" **STRONG BUY**: High opportunity score with favorable risk-reward profile")
            elif opp['Opportunity_Score'] > 4:
                st.info(f" **MONITOR**: Moderate opportunity with some attractive characteristics")
            else:
                st.warning(f" **AVOID**: Low opportunity score with significant challenges")

else:  # Investment Thesis
    st.header(" Comprehensive Investment Thesis")
    
    # Executive summary
    st.markdown('<div class="insight-highlight">', unsafe_allow_html=True)
    st.markdown("##  Executive Summary")
    st.markdown(f"""
    **Market:** SA SMME FinTech • **TAM:** R{market_fundamentals.iloc[2]['Total_Addressable_Market_ZAR_BN']:.1f}BN • **Growth:** {market_fundamentals.iloc[2]['YoY_Growth_Rate']*100:.0f}% YoY
    
    **Thesis:** The SA SMME FinTech sector presents a compelling investment opportunity driven by accelerating digital adoption, 
    strong unit economics, and significant addressable market. While regulatory barriers exist, the growth trajectory 
    and market dynamics support attractive risk-adjusted returns for disciplined investors.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Investment thesis pillars
    st.subheader(" Investment Thesis Pillars")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("###  Growth Drivers")
        st.markdown("""
        - **Digital Adoption:** 18% → 26% (2024-2026)
        - **SMME Digitization:** R2.8BN TAM
        - **Market Consolidation:** Fragmented competitive landscape
        - **Regulatory Tailwinds:** IFWG support for innovation
        """)
    
    with col2:
        st.markdown("###  Financial Attractiveness")
        st.markdown("""
        - **Valuation Multiples:** 9.4x sector average
        - **Gross Margins:** 73% average
        - **Funding Momentum:** R220M quarterly
        - **Exit Potential:** Strategic acquirers active
        """)
    
    with col3:
        st.markdown("###  Risk Mitigation")
        st.markdown("""
        - **Diversified Exposure:** Multiple subsectors
        - **Revenue Model:** Recurring SaaS revenue
        - **Market Timing:** Early growth phase
        - **Regular Monitoring:** Quarterly KPI tracking
        """)
    
    # Investment strategy
    st.subheader(" Recommended Investment Strategy")
    
    strategy_data = pd.DataFrame({
        'Strategy': ['Platform + Specialty', 'Growth-Stage Focus', 'Sector Diversification', 'Value Add'],
        'Description': [
            'Cornerstone investment in platform + 2-3 specialty bets',
            'Target companies with 30-70% YoY growth',
            'Spread across payments, cashflow, lending',
            'Active portfolio management with operational support'
        ],
        'Allocation': ['60%', '30%', '10%', 'Ongoing'],
        'Target_IRR': ['35%+', '40%+', '30%+', 'N/A']
    })
    
    for _, strategy in strategy_data.iterrows():
        col1, col2, col3, col4 = st.columns([2, 4, 1, 1])
        with col1:
            st.markdown(f"**{strategy['Strategy']}**")
        with col2:
            st.markdown(strategy['Description'])
        with col3:
            st.markdown(f"`{strategy['Allocation']}`")
        with col4:
            st.markdown(f"`{strategy['Target_IRR']}`")
        st.divider()
    
    # Return projections
    st.subheader(" Return Projections & Scenarios")
    
    scenarios = pd.DataFrame({
        'Scenario': ['Bull Case', 'Base Case', 'Bear Case'],
        'Description': [
            'Accelerated digital adoption + favorable regulation',
            'Current growth trends continue',
            'Economic downturn + regulatory tightening'
        ],
        '5Y_CAGR': [55, 45, 25],
        'Valuation_Multiple': [12, 9, 6],
        'Expected_IRR': [48, 35, 18],
        'Probability': [30, 50, 20]
    })
    
    fig = go.Figure(data=[
        go.Bar(name='5Y CAGR (%)', x=scenarios['Scenario'], y=scenarios['5Y_CAGR'],
               marker_color=['#10B981', '#3B82F6', '#EF4444']),
        go.Bar(name='Expected IRR (%)', x=scenarios['Scenario'], y=scenarios['Expected_IRR'],
               marker_color=['#059669', '#1D4ED8', '#DC2626'])
    ])
    
    fig.update_layout(
        title='Return Projections by Scenario',
        xaxis_title="Scenario",
        yaxis_title="Percentage (%)",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio construction
    st.subheader(" Recommended Portfolio Construction")
    
    portfolio_allocation = pd.DataFrame({
        'Sector': ['Cashflow Management', 'Embedded Finance', 'Business Banking', 
                  'Payments Infrastructure', 'Lending & Credit'],
        'Allocation': [30, 25, 20, 15, 10],
        'Investment_Size': [45, 37.5, 30, 22.5, 15],
        'Stage': ['Growth', 'Early Growth', 'Growth', 'Established', 'Growth'],
        'Hold_Period': ['3-5 years', '5-7 years', '4-6 years', '2-4 years', '4-6 years']
    })
    
    fig = px.pie(
        portfolio_allocation,
        values='Allocation',
        names='Sector',
        title='Recommended Portfolio Allocation (R150M Fund)',
        color_discrete_sequence=px.colors.sequential.Teal
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Final recommendation
    st.markdown('<div class="insight-highlight">', unsafe_allow_html=True)
    st.markdown("##  Final Recommendation")
    st.markdown(f"""
    **Recommendation: INVEST with R{150}M allocation**
    
    - **Target Holdings:** 6-8 companies across cashflow management, embedded finance, and business banking
    - **Check Size:** R15-25M per investment
    - **Hold Period:** 4-6 years average
    - **Target IRR:** {min_irr}%+ (Base Case: 35%)
    - **Risk Mitigation:** Staged funding, board seats, operational support
    
    *The SA SMME FinTech sector offers attractive risk-adjusted returns with multiple exit pathways 
    and significant growth tailwinds over a {investment_horizon}-year investment horizon.*
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 2rem;'>
    <h3> Data Sources & Methodology</h3>
    <p><strong>Primary Sources:</strong> Company financials, Crunchbase, Disrupt Africa, SARB, Stats SA • 
    <strong>Time Period:</strong> 2022-Q1 to 2024-Q2 • 
    <strong>Analysis Date:</strong> August 2024</p>
    <p><strong>Disclaimer:</strong> This is a conceptual project composed with real and synthetic data for demonstration purpose only. Names of companies are fictitious and do not represent or alias any legally registered entity within the republic or abroad. If my work impress you, reach-out for my services, I'm available for consulting and employment.</p>
    <small>Econometric Dashboard • Prepared for market makers and VCs</small>
</div>
""", unsafe_allow_html=True)

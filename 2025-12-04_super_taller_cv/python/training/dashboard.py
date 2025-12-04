"""
Interactive Dashboard for Model Metrics
Subsystem 5: Training and Model Comparison

Interactive Streamlit dashboard to visualize and compare model performance.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="ML Model Comparison Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


class DashboardData:
    """Load and manage dashboard data"""
    
    def __init__(self, metrics_dir):
        self.metrics_dir = metrics_dir
        self.models_data = {}
        self.comparison_df = None
        
    def load_metrics(self):
        """Load all model metrics"""
        json_files = glob.glob(os.path.join(self.metrics_dir, '*_metrics.json'))
        
        for json_file in json_files:
            model_name = os.path.basename(json_file).replace('_metrics.json', '')
            
            with open(json_file, 'r') as f:
                self.models_data[model_name] = json.load(f)
        
        return len(self.models_data) > 0
    
    def create_comparison_dataframe(self):
        """Create DataFrame for comparison"""
        comparison_data = []
        
        for model_name, metrics in self.models_data.items():
            if 'test' in metrics:
                test_metrics = metrics['test']
                
                row = {
                    'Model': model_name.upper().replace('_', ' '),
                    'Accuracy': test_metrics.get('accuracy', 0),
                    'Precision': test_metrics.get('precision', 0),
                    'Recall': test_metrics.get('recall', 0),
                    'AUC': test_metrics.get('auc', 0),
                    'Loss': test_metrics.get('loss', 0)
                }
                
                comparison_data.append(row)
        
        self.comparison_df = pd.DataFrame(comparison_data)
        self.comparison_df = self.comparison_df.sort_values('Accuracy', ascending=False)
        
        return self.comparison_df


def plot_metrics_comparison(df):
    """Create interactive bar chart for metrics comparison"""
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metrics,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set3
    
    for idx, metric in enumerate(metrics):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        fig.add_trace(
            go.Bar(
                x=df['Model'],
                y=df[metric],
                name=metric,
                marker_color=colors[idx],
                text=df[metric].round(4),
                textposition='outside',
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(tickangle=-45, row=row, col=col)
        fig.update_yaxes(range=[0, 1.0], row=row, col=col)
    
    fig.update_layout(
        title_text="Model Performance Metrics Comparison",
        title_font_size=20,
        height=700,
        showlegend=False
    )
    
    return fig


def plot_radar_chart(df):
    """Create interactive radar chart"""
    
    categories = ['Accuracy', 'Precision', 'Recall', 'AUC']
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[cat] for cat in categories]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=row['Model'],
            line_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Model Performance Radar Chart",
        title_font_size=20,
        height=600
    )
    
    return fig


def plot_precision_recall_scatter(df):
    """Create precision vs recall scatter plot"""
    
    fig = px.scatter(
        df,
        x='Recall',
        y='Precision',
        text='Model',
        size=[100] * len(df),
        color='Model',
        title='Precision vs Recall',
        range_x=[0, 1],
        range_y=[0, 1]
    )
    
    # Add diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            showlegend=False,
            name='Reference'
        )
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=600, title_font_size=20)
    
    return fig


def plot_loss_comparison(df):
    """Create loss comparison chart"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Model'],
        y=df['Loss'],
        marker_color='indianred',
        text=df['Loss'].round(4),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Model Loss Comparison",
        title_font_size=20,
        xaxis_title="Model",
        yaxis_title="Test Loss",
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig


def plot_heatmap(df):
    """Create metrics heatmap"""
    
    metrics_data = df[['Model', 'Accuracy', 'Precision', 'Recall', 'AUC']].set_index('Model').T
    
    fig = go.Figure(data=go.Heatmap(
        z=metrics_data.values,
        x=metrics_data.columns,
        y=metrics_data.index,
        colorscale='YlGnBu',
        text=metrics_data.values.round(4),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Score")
    ))
    
    fig.update_layout(
        title="Metrics Heatmap",
        title_font_size=20,
        height=400,
        xaxis_title="Model",
        yaxis_title="Metric"
    )
    
    return fig


def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 ML Model Comparison Dashboard</h1>', 
               unsafe_allow_html=True)
    st.markdown("### Subsystem 5: Training and Model Comparison")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Dashboard Settings")
    st.sidebar.markdown("---")
    
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    
    # Check if directory exists
    if not os.path.exists(metrics_dir):
        st.error(f"❌ Metrics directory not found: {metrics_dir}")
        st.info("Please train models first before running the dashboard.")
        return
    
    # Load data
    @st.cache_data
    def load_dashboard_data():
        data_loader = DashboardData(metrics_dir)
        if data_loader.load_metrics():
            df = data_loader.create_comparison_dataframe()
            return data_loader, df
        return None, None
    
    data_loader, df = load_dashboard_data()
    
    if df is None or len(df) == 0:
        st.warning("⚠️ No model metrics found. Please train models first.")
        st.info("""
        **To generate metrics:**
        1. Run `python python/training/cnn_from_scratch.py`
        2. Run `python python/training/fine_tuning.py`
        3. Refresh this dashboard
        """)
        return
    
    # Sidebar filters
    st.sidebar.subheader("📊 Model Selection")
    selected_models = st.sidebar.multiselect(
        "Select models to compare:",
        options=df['Model'].tolist(),
        default=df['Model'].tolist()
    )
    
    if not selected_models:
        st.warning("Please select at least one model.")
        return
    
    # Filter dataframe
    filtered_df = df[df['Model'].isin(selected_models)]
    
    # Display statistics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Quick Stats")
    st.sidebar.metric("Total Models", len(df))
    st.sidebar.metric("Best Accuracy", f"{df['Accuracy'].max():.4f}")
    st.sidebar.metric("Avg Accuracy", f"{df['Accuracy'].mean():.4f}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "📈 Detailed Metrics", 
        "🎯 Comparisons",
        "📄 Raw Data"
    ])
    
    with tab1:
        st.header("Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        best_model = df.loc[df['Accuracy'].idxmax()]
        
        with col1:
            st.metric(
                "🏆 Best Model",
                best_model['Model'],
                f"{best_model['Accuracy']:.4f}"
            )
        
        with col2:
            st.metric(
                "📊 Avg Accuracy",
                f"{df['Accuracy'].mean():.4f}",
                f"±{df['Accuracy'].std():.4f}"
            )
        
        with col3:
            st.metric(
                "🎯 Avg Precision",
                f"{df['Precision'].mean():.4f}",
                f"±{df['Precision'].std():.4f}"
            )
        
        with col4:
            st.metric(
                "🔄 Avg Recall",
                f"{df['Recall'].mean():.4f}",
                f"±{df['Recall'].std():.4f}"
            )
        
        st.markdown("---")
        
        # Radar chart
        st.subheader("Performance Radar Chart")
        fig_radar = plot_radar_chart(filtered_df)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Heatmap
        st.subheader("Metrics Heatmap")
        fig_heatmap = plot_heatmap(filtered_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        st.header("Detailed Metrics")
        
        # Metrics comparison
        fig_metrics = plot_metrics_comparison(filtered_df)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        st.markdown("---")
        
        # Loss comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Loss Comparison")
            fig_loss = plot_loss_comparison(filtered_df)
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            st.subheader("Precision vs Recall")
            fig_pr = plot_precision_recall_scatter(filtered_df)
            st.plotly_chart(fig_pr, use_container_width=True)
    
    with tab3:
        st.header("Model Comparisons")
        
        # Select models to compare
        st.subheader("Select Two Models to Compare")
        col1, col2 = st.columns(2)
        
        with col1:
            model1 = st.selectbox("Model 1", df['Model'].tolist(), index=0)
        
        with col2:
            model2 = st.selectbox("Model 2", df['Model'].tolist(), 
                                 index=min(1, len(df)-1))
        
        if model1 != model2:
            model1_data = df[df['Model'] == model1].iloc[0]
            model2_data = df[df['Model'] == model2].iloc[0]
            
            st.markdown("---")
            
            # Comparison metrics
            col1, col2, col3, col4 = st.columns(4)
            
            metrics_list = ['Accuracy', 'Precision', 'Recall', 'AUC']
            
            for col, metric in zip([col1, col2, col3, col4], metrics_list):
                with col:
                    st.metric(
                        f"{metric}",
                        f"{model1_data[metric]:.4f}",
                        f"{(model1_data[metric] - model2_data[metric]):.4f}"
                    )
            
            # Side-by-side comparison
            st.markdown("---")
            st.subheader("Side-by-Side Comparison")
            
            comparison_data = {
                'Metric': metrics_list,
                model1: [model1_data[m] for m in metrics_list],
                model2: [model2_data[m] for m in metrics_list]
            }
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name=model1,
                x=metrics_list,
                y=[model1_data[m] for m in metrics_list],
                text=[f"{model1_data[m]:.4f}" for m in metrics_list],
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                name=model2,
                x=metrics_list,
                y=[model2_data[m] for m in metrics_list],
                text=[f"{model2_data[m]:.4f}" for m in metrics_list],
                textposition='outside'
            ))
            
            fig.update_layout(
                barmode='group',
                title=f"{model1} vs {model2}",
                yaxis=dict(range=[0, 1.0]),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select two different models to compare.")
    
    with tab4:
        st.header("Raw Data")
        
        st.subheader("Comparison Table")
        st.dataframe(
            filtered_df.style.highlight_max(axis=0, 
                                           subset=['Accuracy', 'Precision', 'Recall', 'AUC'],
                                           color='lightgreen')
                         .highlight_min(axis=0,
                                       subset=['Loss'],
                                       color='lightgreen')
                         .format({'Accuracy': '{:.4f}',
                                 'Precision': '{:.4f}',
                                 'Recall': '{:.4f}',
                                 'AUC': '{:.4f}',
                                 'Loss': '{:.4f}'}),
            use_container_width=True
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="model_comparison.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Detailed metrics for each model
        st.subheader("Detailed Model Information")
        selected_model_detail = st.selectbox(
            "Select model for detailed view:",
            filtered_df['Model'].tolist()
        )
        
        if selected_model_detail:
            model_key = selected_model_detail.lower().replace(' ', '_')
            
            if model_key in data_loader.models_data:
                st.json(data_loader.models_data[model_key])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🤖 ML Model Comparison Dashboard | Subsystem 5 | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

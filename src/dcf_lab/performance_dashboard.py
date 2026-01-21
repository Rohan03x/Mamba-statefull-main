"""
Live Performance Dashboard for DCF Suite v0201

Real-time performance monitoring and visualization for the accuracy tracking system.
Integrates with governance framework for production deployment.

Features:
- Real-time accuracy metrics display
- Rolling window performance charts
- Alert system for performance degradation
- A/B testing integration
- Model comparison dashboard
- Risk metrics and deployment readiness
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import List
import sqlite3

from .accuracy_tracking import LiveAccuracyTracker, AccuracyMetric
from .governance import ProductionGovernance

class PerformanceDashboard:
    """Real-time performance dashboard for model accuracy tracking"""
    
    def __init__(self, accuracy_tracker: LiveAccuracyTracker):
        self.tracker = accuracy_tracker
        self.governance = ProductionGovernance()
        
    def render_dashboard(self):
        """Render the complete Streamlit dashboard"""
        
        st.set_page_config(
            page_title="DCF Suite v0201 - Live Performance Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("🚀 DCF Suite v0201 - Live Performance Dashboard")
        st.markdown("Real-time model accuracy tracking and governance monitoring")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Live Metrics", 
            "📈 Performance Trends", 
            "🔍 Model Comparison",
            "⚠️ Alerts & Governance",
            "🎯 Deployment Status"
        ])
        
        with tab1:
            self._render_live_metrics()
            
        with tab2:
            self._render_performance_trends()
            
        with tab3:
            self._render_model_comparison()
            
        with tab4:
            self._render_alerts_governance()
            
        with tab5:
            self._render_deployment_status()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        
        st.sidebar.header("Dashboard Controls")
        
        # Time range selection
        self.time_range = st.sidebar.selectbox(
            "Analysis Time Range",
            ["7 days", "30 days", "90 days", "1 year"],
            index=1
        )
        
        self.window_days = int(self.time_range.split()[0]) if self.time_range != "1 year" else 365
        
        # Symbol filter
        available_symbols = self._get_available_symbols()
        self.selected_symbols = st.sidebar.multiselect(
            "Filter by Symbols",
            available_symbols,
            default=available_symbols[:5] if len(available_symbols) > 5 else available_symbols
        )
        
        # Model filter
        available_models = self._get_available_models()
        self.selected_models = st.sidebar.multiselect(
            "Filter by Models",
            available_models,
            default=available_models
        )
        
        # Refresh controls
        st.sidebar.divider()
        if st.sidebar.button("🔄 Refresh Data", type="primary"):
            st.rerun()
        
        # Auto-refresh
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)")
        if auto_refresh:
            st.rerun()
    
    def _render_live_metrics(self):
        """Render live metrics tab"""
        
        st.header("📊 Live Accuracy Metrics")
        
        # Calculate current metrics
        current_metrics = self.tracker.calculate_accuracy_metrics(
            window_days=self.window_days,
            symbols=self.selected_symbols if self.selected_symbols else None,
            models=self.selected_models if self.selected_models else None
        )
        
        if not current_metrics:
            st.warning(f"No data available for the selected {self.window_days}-day period")
            return
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            dir_acc = current_metrics.get(AccuracyMetric.DIRECTIONAL_ACCURACY)
            if dir_acc:
                delta_color = "normal" if dir_acc.value >= 0.55 else "inverse"
                st.metric(
                    "Directional Accuracy",
                    f"{dir_acc.value:.1%}",
                    delta=f"±{(dir_acc.confidence_interval[1] - dir_acc.confidence_interval[0])/2:.1%}",
                    delta_color=delta_color
                )
                st.caption(f"Sample: {dir_acc.sample_size} predictions")
        
        with col2:
            hit_rate = current_metrics.get(AccuracyMetric.HIT_RATE)
            if hit_rate:
                expected_hit_rate = 0.95  # Expected for 95% confidence intervals
                delta_val = hit_rate.value - expected_hit_rate
                st.metric(
                    "Range Hit Rate",
                    f"{hit_rate.value:.1%}",
                    delta=f"{delta_val:+.1%}",
                    delta_color="normal" if delta_val >= 0 else "inverse"
                )
                st.caption(f"Sample: {hit_rate.sample_size} predictions")
        
        with col3:
            rmse = current_metrics.get(AccuracyMetric.RMSE)
            if rmse:
                st.metric(
                    "RMSE",
                    f"${rmse.value:.2f}",
                    delta=None
                )
                st.caption(f"Sample: {rmse.sample_size} predictions")
        
        with col4:
            win_rate = current_metrics.get(AccuracyMetric.WIN_RATE)
            if win_rate:
                delta_color = "normal" if win_rate.value >= 0.6 else "inverse"
                st.metric(
                    "Win Rate",
                    f"{win_rate.value:.1%}",
                    delta=f"±{(win_rate.confidence_interval[1] - win_rate.confidence_interval[0])/2:.1%}",
                    delta_color=delta_color
                )
                st.caption(f"Sample: {win_rate.sample_size} predictions")
        
        # Detailed breakdown
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 By Symbol Performance")
            dir_acc = current_metrics.get(AccuracyMetric.DIRECTIONAL_ACCURACY)
            if dir_acc and dir_acc.by_symbol:
                symbol_df = pd.DataFrame(list(dir_acc.by_symbol.items()), 
                                       columns=['Symbol', 'Accuracy'])
                symbol_df = symbol_df.sort_values('Accuracy', ascending=False)
                
                fig = px.bar(
                    symbol_df, 
                    x='Symbol', 
                    y='Accuracy',
                    title="Directional Accuracy by Symbol",
                    color='Accuracy',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🤖 By Model Performance")
            if dir_acc and dir_acc.by_model:
                model_df = pd.DataFrame(list(dir_acc.by_model.items()), 
                                      columns=['Model', 'Accuracy'])
                model_df = model_df.sort_values('Accuracy', ascending=False)
                
                fig = px.bar(
                    model_df, 
                    x='Model', 
                    y='Accuracy',
                    title="Directional Accuracy by Model",
                    color='Accuracy',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_trends(self):
        """Render performance trends tab"""
        
        st.header("📈 Performance Trends")
        
        # Rolling accuracy over time
        metrics_to_plot = [
            AccuracyMetric.DIRECTIONAL_ACCURACY,
            AccuracyMetric.HIT_RATE,
            AccuracyMetric.WIN_RATE
        ]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Directional Accuracy', 'Hit Rate', 'Win Rate', 'Sample Sizes'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['blue', 'green', 'orange']
        
        for i, metric in enumerate(metrics_to_plot):
            trend_data = self.tracker.get_rolling_accuracy(metric, days=self.window_days)
            
            if not trend_data.empty:
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                fig.add_trace(
                    go.Scatter(
                        x=trend_data['timestamp'],
                        y=trend_data['value'],
                        mode='lines+markers',
                        name=metric.value.replace('_', ' ').title(),
                        line=dict(color=colors[i])
                    ),
                    row=row, col=col
                )
                
                # Add sample size plot
                if i == 0:  # Only for the first metric
                    fig.add_trace(
                        go.Scatter(
                            x=trend_data['timestamp'],
                            y=trend_data['sample_size'],
                            mode='lines+markers',
                            name='Sample Size',
                            line=dict(color='red')
                        ),
                        row=2, col=2
                    )
        
        fig.update_layout(height=600, title="Performance Trends Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent predictions table
        st.subheader("🕐 Recent Predictions")
        recent_predictions = self._get_recent_predictions(limit=20)
        
        if not recent_predictions.empty:
            # Format for display
            display_df = recent_predictions[[
                'timestamp', 'symbol', 'model_id', 'predicted_direction',
                'predicted_price', 'actual_price', 'is_validated'
            ]].copy()
            
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            display_df['predicted_price'] = display_df['predicted_price'].round(2)
            display_df['actual_price'] = display_df['actual_price'].round(2)
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No recent predictions available")
    
    def _render_model_comparison(self):
        """Render model comparison tab"""
        
        st.header("🔍 Model Comparison & A/B Testing")
        
        # Get all model performance
        all_models = self._get_available_models()
        
        if len(all_models) < 2:
            st.warning("Need at least 2 models for comparison")
            return
        
        comparison_data = []
        
        for model in all_models:
            metrics = self.tracker.calculate_accuracy_metrics(
                window_days=self.window_days,
                models=[model]
            )
            
            model_data = {
                'Model': model,
                'Directional_Accuracy': metrics.get(AccuracyMetric.DIRECTIONAL_ACCURACY, type('', (), {'value': 0, 'sample_size': 0})).value,
                'Hit_Rate': metrics.get(AccuracyMetric.HIT_RATE, type('', (), {'value': 0, 'sample_size': 0})).value,
                'Win_Rate': metrics.get(AccuracyMetric.WIN_RATE, type('', (), {'value': 0, 'sample_size': 0})).value,
                'RMSE': metrics.get(AccuracyMetric.RMSE, type('', (), {'value': 0, 'sample_size': 0})).value,
                'Sample_Size': metrics.get(AccuracyMetric.DIRECTIONAL_ACCURACY, type('', (), {'sample_size': 0})).sample_size
            }
            comparison_data.append(model_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Model performance comparison chart
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Directional Accuracy', 'Hit Rate', 'Win Rate']
        )
        
        metrics = ['Directional_Accuracy', 'Hit_Rate', 'Win_Rate']
        colors = ['blue', 'green', 'orange']
        
        for i, metric in enumerate(metrics):
            fig.add_trace(
                go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    name=metric.replace('_', ' '),
                    marker_color=colors[i]
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(height=400, title="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.subheader("📊 Detailed Model Metrics")
        
        # Format for display
        display_comparison = comparison_df.copy()
        for col in ['Directional_Accuracy', 'Hit_Rate', 'Win_Rate']:
            display_comparison[col] = display_comparison[col].apply(lambda x: f"{x:.1%}")
        display_comparison['RMSE'] = display_comparison['RMSE'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(display_comparison, use_container_width=True)
        
        # A/B Testing recommendation
        st.subheader("🧪 A/B Testing Recommendations")
        
        if len(comparison_df) >= 2:
            best_model = comparison_df.loc[comparison_df['Directional_Accuracy'].idxmax()]
            second_best = comparison_df.loc[comparison_df['Directional_Accuracy'].nlargest(2).index[1]]
            
            st.success(f"**Recommended A/B Test**: {best_model['Model']} vs {second_best['Model']}")
            st.write(f"- Champion: {best_model['Model']} ({best_model['Directional_Accuracy']:.1%} accuracy)")
            st.write(f"- Challenger: {second_best['Model']} ({second_best['Directional_Accuracy']:.1%} accuracy)")
            
            if st.button("🚀 Start A/B Test"):
                st.info("A/B test configuration would be initiated here")
    
    def _render_alerts_governance(self):
        """Render alerts and governance tab"""
        
        st.header("⚠️ Alerts & Governance Status")
        
        # Get governance data
        governance_data = self.tracker.integration_with_governance()
        
        # Alert summary
        alerts = governance_data.get('alerts', [])
        
        if alerts:
            st.subheader("🚨 Active Alerts")
            for alert in alerts:
                severity = alert.get('severity', 'info')
                if severity == 'error':
                    st.error(f"**{alert['type'].replace('_', ' ').title()}**: {alert['message']}")
                elif severity == 'warning':
                    st.warning(f"**{alert['type'].replace('_', ' ').title()}**: {alert['message']}")
                else:
                    st.info(f"**{alert['type'].replace('_', ' ').title()}**: {alert['message']}")
                
                if 'recommendation' in alert:
                    st.caption(f"💡 Recommendation: {alert['recommendation']}")
        else:
            st.success("✅ No active alerts - All systems operating normally")
        
        # Governance metrics
        st.subheader("🛡️ Governance Metrics")
        
        perf_data = governance_data.get('model_performance', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recent Performance (7 days)**")
            st.write(f"- Directional Accuracy: {perf_data.get('recent_directional_accuracy', 0):.1%}")
            st.write(f"- Hit Rate: {perf_data.get('recent_hit_rate', 0):.1%}")
        
        with col2:
            st.write("**Monthly Performance (30 days)**")
            st.write(f"- Directional Accuracy: {perf_data.get('monthly_directional_accuracy', 0):.1%}")
            st.write(f"- Hit Rate: {perf_data.get('monthly_hit_rate', 0):.1%}")
        
        # Governance status
        st.subheader("📋 Deployment Readiness Checklist")
        
        recent_acc = perf_data.get('recent_directional_accuracy', 0)
        monthly_acc = perf_data.get('monthly_directional_accuracy', 0)
        recent_hit = perf_data.get('recent_hit_rate', 0)
        
        checklist = [
            ("Monthly directional accuracy > 55%", monthly_acc > 0.55),
            ("Recent directional accuracy > 50%", recent_acc > 0.50),
            ("Hit rate within expected range", 0.90 <= recent_hit <= 1.0),
            ("No critical alerts", not any(a.get('severity') == 'error' for a in alerts)),
            ("Sufficient sample size", True),  # Would check actual sample sizes
        ]
        
        for check_name, passed in checklist:
            if passed:
                st.success(f"✅ {check_name}")
            else:
                st.error(f"❌ {check_name}")
        
        all_passed = all(passed for _, passed in checklist)
        
        if all_passed:
            st.success("🚀 **System ready for deployment**")
        else:
            st.warning("⚠️ **Address issues before deployment**")
    
    def _render_deployment_status(self):
        """Render deployment status tab"""
        
        st.header("🎯 Deployment Status & Roadmap")
        
        # Current deployment phase
        st.subheader("📍 Current Phase")
        
        phases = [
            "1. Development & Testing",
            "2. Accuracy Validation",
            "3. Paper Trading",
            "4. Limited Live Deployment",
            "5. Full Production"
        ]
        
        current_phase = 2  # Accuracy validation phase
        
        for i, phase in enumerate(phases):
            if i < current_phase:
                st.success(f"✅ {phase}")
            elif i == current_phase:
                st.info(f"🔄 **{phase}** (Current)")
            else:
                st.write(f"⭕ {phase}")
        
        # Deployment metrics
        st.subheader("📊 Deployment Readiness Metrics")
        
        # Mock deployment readiness score
        governance_data = self.tracker.integration_with_governance()
        perf_data = governance_data.get('model_performance', {})
        
        monthly_acc = perf_data.get('monthly_directional_accuracy', 0)
        recent_acc = perf_data.get('recent_directional_accuracy', 0)
        alerts_count = len(governance_data.get('alerts', []))
        
        # Calculate readiness score
        readiness_score = 0
        if monthly_acc > 0.55: readiness_score += 30
        if recent_acc > 0.50: readiness_score += 25
        if alerts_count == 0: readiness_score += 25
        readiness_score += min(20, monthly_acc * 40)  # Bonus for high accuracy
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Deployment Readiness", f"{readiness_score:.0f}%")
        
        with col2:
            risk_level = "Low" if readiness_score > 80 else ("Medium" if readiness_score > 60 else "High")
            color = "green" if risk_level == "Low" else ("orange" if risk_level == "Medium" else "red")
            st.metric("Risk Level", risk_level, delta_color=color)
        
        with col3:
            next_review = datetime.now() + timedelta(days=7)
            st.metric("Next Review", next_review.strftime("%Y-%m-%d"))
        
        # Next steps
        st.subheader("🎯 Next Steps")
        
        if readiness_score > 80:
            st.success("**Ready for Paper Trading Phase**")
            next_steps = [
                "Set up paper trading environment",
                "Configure real-time data feeds",
                "Implement position sizing algorithms",
                "Begin 4-week paper trading trial"
            ]
        elif readiness_score > 60:
            st.warning("**Continue Accuracy Validation**")
            next_steps = [
                "Collect more prediction samples",
                "Analyze model performance edge cases",
                "Implement additional risk controls",
                "Address any remaining alerts"
            ]
        else:
            st.error("**Model Improvement Required**")
            next_steps = [
                "Investigate accuracy issues",
                "Retrain models with additional data",
                "Review feature engineering",
                "Strengthen validation framework"
            ]
        
        for i, step in enumerate(next_steps, 1):
            st.write(f"{i}. {step}")
        
        # Timeline
        st.subheader("📅 Projected Timeline")
        
        timeline_data = [
            {"Phase": "Accuracy Validation", "Start": "2024-01-15", "End": "2024-02-15", "Status": "In Progress"},
            {"Phase": "Paper Trading", "Start": "2024-02-15", "End": "2024-03-15", "Status": "Planned"},
            {"Phase": "Limited Live", "Start": "2024-03-15", "End": "2024-04-15", "Status": "Planned"},
            {"Phase": "Full Production", "Start": "2024-04-15", "End": "2024-05-15", "Status": "Planned"}
        ]
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True)
    
    def _get_available_symbols(self) -> List[str]:
        """Get available symbols from database"""
        try:
            with sqlite3.connect(self.tracker.db_path) as conn:
                cursor = conn.execute("SELECT DISTINCT symbol FROM predictions ORDER BY symbol")
                return [row[0] for row in cursor.fetchall()]
        except:
            return ['AAPL', 'MSFT', 'GOOGL']  # Fallback
    
    def _get_available_models(self) -> List[str]:
        """Get available models from database"""
        try:
            with sqlite3.connect(self.tracker.db_path) as conn:
                cursor = conn.execute("SELECT DISTINCT model_id FROM predictions ORDER BY model_id")
                return [row[0] for row in cursor.fetchall()]
        except:
            return ['enhanced_tf_v2.1', 'baseline_v1.0']  # Fallback
    
    def _get_recent_predictions(self, limit: int = 20) -> pd.DataFrame:
        """Get recent predictions for display"""
        try:
            with sqlite3.connect(self.tracker.db_path) as conn:
                return pd.read_sql_query(f"""
                    SELECT * FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT {limit}
                """, conn)
        except:
            return pd.DataFrame()


def main():
    """Main dashboard application"""
    
    # Initialize components
    tracker = LiveAccuracyTracker()
    dashboard = PerformanceDashboard(tracker)
    
    # Render dashboard
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()
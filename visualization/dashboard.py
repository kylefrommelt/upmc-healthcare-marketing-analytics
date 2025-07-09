"""
Healthcare Marketing Dashboard for UPMC Analytics
Comprehensive visualization dashboard for marketing analytics insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

class HealthcareMarketingDashboard:
    """
    Interactive dashboard for healthcare marketing analytics
    Demonstrates advanced data visualization and business intelligence skills
    """
    
    def __init__(self):
        pass
        
    def show_overview(self, results):
        """Display executive overview dashboard"""
        st.header("📊 Executive Overview")
        
        # Extract key metrics
        campaign_data = results['campaign_data']
        patient_data = results['patient_data']
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_budget = campaign_data['budget'].sum()
            st.metric("Total Marketing Budget", f"${total_budget:,.0f}")
        
        with col2:
            total_conversions = campaign_data['conversions'].sum()
            st.metric("Total Patient Acquisitions", f"{total_conversions:,.0f}")
        
        with col3:
            avg_patient_value = patient_data['patient_value'].mean()
            st.metric("Avg Patient Value", f"${avg_patient_value:,.0f}")
        
        with col4:
            total_roi = ((total_conversions * avg_patient_value) - total_budget) / total_budget
            st.metric("Overall ROI", f"{total_roi:.1%}")
        
        # Channel performance overview
        st.subheader("Channel Performance Overview")
        
        channel_performance = campaign_data.groupby('channel').agg({
            'budget': 'sum',
            'conversions': 'sum',
            'impressions': 'sum',
            'clicks': 'sum'
        }).reset_index()
        
        channel_performance['cost_per_conversion'] = channel_performance['budget'] / channel_performance['conversions']
        channel_performance['roi'] = ((channel_performance['conversions'] * avg_patient_value) - channel_performance['budget']) / channel_performance['budget']
        
        # Channel ROI chart
        fig_roi = px.bar(
            channel_performance, 
            x='channel', 
            y='roi',
            title="ROI by Marketing Channel",
            labels={'roi': 'ROI (%)', 'channel': 'Marketing Channel'},
            color='roi',
            color_continuous_scale='RdYlGn'
        )
        fig_roi.update_layout(showlegend=False)
        st.plotly_chart(fig_roi, use_container_width=True)
        
        # Budget allocation vs performance
        col1, col2 = st.columns(2)
        
        with col1:
            fig_budget = px.pie(
                channel_performance, 
                values='budget', 
                names='channel',
                title="Budget Allocation by Channel"
            )
            st.plotly_chart(fig_budget, use_container_width=True)
        
        with col2:
            fig_conversions = px.pie(
                channel_performance, 
                values='conversions', 
                names='channel',
                title="Patient Acquisitions by Channel"
            )
            st.plotly_chart(fig_conversions, use_container_width=True)
        
        # Specialty analysis
        st.subheader("Healthcare Specialty Analysis")
        
        specialty_performance = campaign_data.groupby('specialty').agg({
            'budget': 'sum',
            'conversions': 'sum'
        }).reset_index()
        
        specialty_performance['cost_per_conversion'] = specialty_performance['budget'] / specialty_performance['conversions']
        
        fig_specialty = px.bar(
            specialty_performance, 
            x='specialty', 
            y='cost_per_conversion',
            title="Cost per Patient Acquisition by Specialty",
            labels={'cost_per_conversion': 'Cost per Conversion ($)', 'specialty': 'Healthcare Specialty'}
        )
        fig_specialty.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_specialty, use_container_width=True)
        
        # Performance trends
        st.subheader("Performance Trends")
        
        # Monthly trends
        campaign_data['month'] = pd.to_datetime(campaign_data['start_date']).dt.to_period('M')
        monthly_trends = campaign_data.groupby('month').agg({
            'budget': 'sum',
            'conversions': 'sum',
            'impressions': 'sum'
        }).reset_index()
        
        monthly_trends['month'] = monthly_trends['month'].astype(str)
        monthly_trends['cost_per_conversion'] = monthly_trends['budget'] / monthly_trends['conversions']
        
        fig_trends = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Budget vs Conversions', 'Monthly Cost per Conversion'),
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Top subplot - Budget vs Conversions
        fig_trends.add_trace(
            go.Scatter(x=monthly_trends['month'], y=monthly_trends['budget'], 
                      name='Budget', line=dict(color='blue')),
            row=1, col=1
        )
        fig_trends.add_trace(
            go.Scatter(x=monthly_trends['month'], y=monthly_trends['conversions'], 
                      name='Conversions', line=dict(color='red')),
            row=1, col=1, secondary_y=True
        )
        
        # Bottom subplot - Cost per Conversion
        fig_trends.add_trace(
            go.Scatter(x=monthly_trends['month'], y=monthly_trends['cost_per_conversion'], 
                      name='Cost per Conversion', line=dict(color='green')),
            row=2, col=1
        )
        
        fig_trends.update_layout(height=600, title_text="Marketing Performance Trends")
        st.plotly_chart(fig_trends, use_container_width=True)
        
    def show_media_mix_analysis(self, media_mix_results):
        """Display media mix modeling results"""
        st.header("📺 Media Mix Modeling Analysis")
        
        # Model performance
        st.subheader("Model Performance")
        
        performance = media_mix_results['model_performance']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training R²", f"{performance['train_r2']:.3f}")
        with col2:
            st.metric("Test R²", f"{performance['test_r2']:.3f}")
        with col3:
            st.metric("Training MAE", f"{performance['train_mae']:.2f}")
        with col4:
            st.metric("Test MAE", f"{performance['test_mae']:.2f}")
        
        # Channel contributions
        st.subheader("Channel Contribution Analysis")
        
        contributions = media_mix_results['channel_contributions']
        
        if contributions:
            contrib_df = pd.DataFrame([
                {
                    'channel': channel,
                    'contribution_percentage': data['contribution_percentage'],
                    'total_contribution': data['total_contribution']
                }
                for channel, data in contributions.items()
            ])
            
            # Contribution pie chart
            fig_contrib = px.pie(
                contrib_df, 
                values='contribution_percentage', 
                names='channel',
                title="Channel Contribution to Total Conversions"
            )
            st.plotly_chart(fig_contrib, use_container_width=True)
            
            # Contribution table
            st.subheader("Detailed Channel Contributions")
            st.dataframe(contrib_df.sort_values('contribution_percentage', ascending=False))
        
        # Elasticity analysis
        st.subheader("Marketing Elasticity Analysis")
        
        elasticity = media_mix_results['elasticity_analysis']
        
        if elasticity:
            elasticity_df = pd.DataFrame([
                {
                    'channel': channel,
                    'elasticity': data['elasticity'],
                    'interpretation': data['interpretation']
                }
                for channel, data in elasticity.items()
            ])
            
            # Elasticity bar chart
            fig_elasticity = px.bar(
                elasticity_df, 
                x='channel', 
                y='elasticity',
                title="Marketing Elasticity by Channel",
                labels={'elasticity': 'Elasticity', 'channel': 'Marketing Channel'},
                color='elasticity',
                color_continuous_scale='RdYlGn'
            )
            fig_elasticity.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_elasticity, use_container_width=True)
            
            # Elasticity interpretation
            st.subheader("Elasticity Interpretation")
            for _, row in elasticity_df.iterrows():
                st.write(f"**{row['channel']}**: {row['interpretation']}")
        
        # Optimization recommendations
        st.subheader("Budget Optimization Recommendations")
        
        recommendations = media_mix_results['optimization_recommendations']
        
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            
            # Color code recommendations
            def get_recommendation_color(rec):
                if 'INCREASE' in rec:
                    return '🟢'
                elif 'DECREASE' in rec:
                    return '🔴'
                elif 'MAINTAIN' in rec:
                    return '🟡'
                else:
                    return '🔵'
            
            rec_df['priority'] = rec_df['recommendation'].apply(get_recommendation_color)
            
            # Display recommendations
            for _, row in rec_df.iterrows():
                st.write(f"{row['priority']} **{row['channel']}**: {row['recommendation']}")
                st.write(f"   - Current Budget: ${row['current_budget']:,.0f}")
                st.write(f"   - Cost per Conversion: ${row['cost_per_conversion']:.2f}")
                st.write(f"   - Elasticity: {row['elasticity']:.2f}")
                st.write(f"   - Contribution: {row['contribution_percentage']:.1f}%")
                st.write("")
        
    def show_attribution_analysis(self, attribution_results):
        """Display attribution analysis results"""
        st.header("🎯 Multi-Touch Attribution Analysis")
        
        # Model performance
        st.subheader("Attribution Model Performance")
        
        performance = attribution_results['model_performance']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training AUC", f"{performance['train_auc']:.3f}")
        with col2:
            st.metric("Test AUC", f"{performance['test_auc']:.3f}")
        with col3:
            st.metric("Training Accuracy", f"{performance['train_accuracy']:.3f}")
        with col4:
            st.metric("Test Accuracy", f"{performance['test_accuracy']:.3f}")
        
        # Channel attribution comparison
        st.subheader("Attribution Model Comparison")
        
        channel_attribution = attribution_results['channel_attribution']
        attribution_comparison = channel_attribution['attribution_comparison']
        
        # Multi-touch vs Last-touch attribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_multi_touch = px.bar(
                attribution_comparison, 
                x='channel', 
                y='total_attributed_value',
                title="Multi-Touch Attribution Value",
                labels={'total_attributed_value': 'Attributed Value ($)', 'channel': 'Channel'}
            )
            fig_multi_touch.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_multi_touch, use_container_width=True)
        
        with col2:
            fig_last_touch = px.bar(
                attribution_comparison, 
                x='channel', 
                y='last_touch_value',
                title="Last-Touch Attribution Value",
                labels={'last_touch_value': 'Last-Touch Value ($)', 'channel': 'Channel'}
            )
            fig_last_touch.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_last_touch, use_container_width=True)
        
        # Attribution lift analysis
        st.subheader("Attribution Lift Analysis")
        
        # Remove infinite values for visualization
        attribution_comparison_clean = attribution_comparison.copy()
        attribution_comparison_clean['attribution_lift'] = attribution_comparison_clean['attribution_lift'].replace([np.inf, -np.inf], 0)
        
        fig_lift = px.bar(
            attribution_comparison_clean, 
            x='channel', 
            y='attribution_lift',
            title="Attribution Lift: Multi-Touch vs Last-Touch (%)",
            labels={'attribution_lift': 'Attribution Lift (%)', 'channel': 'Channel'},
            color='attribution_lift',
            color_continuous_scale='RdYlGn'
        )
        fig_lift.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_lift, use_container_width=True)
        
        # Journey analysis
        st.subheader("Patient Journey Analysis")
        
        journey_analysis = attribution_results['journey_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Journey Length", f"{journey_analysis['avg_journey_length']:.1f} touchpoints")
            st.metric("Average Time Between Touchpoints", f"{journey_analysis['avg_time_between_touchpoints']:.1f} days")
        
        with col2:
            # Journey length distribution
            journey_dist = journey_analysis['journey_length_distribution']
            journey_dist_df = pd.DataFrame(list(journey_dist.items()), columns=['Journey Length', 'Count'])
            
            fig_journey_dist = px.bar(
                journey_dist_df, 
                x='Journey Length', 
                y='Count',
                title="Distribution of Patient Journey Lengths"
            )
            st.plotly_chart(fig_journey_dist, use_container_width=True)
        
        # Common journey patterns
        st.subheader("Most Common Patient Journey Patterns")
        
        journey_patterns = journey_analysis['common_journey_patterns']
        
        if journey_patterns:
            for i, (pattern, count) in enumerate(journey_patterns.items(), 1):
                st.write(f"{i}. **{pattern}** ({count} patients)")
        
        # Channel position analysis
        st.subheader("Channel Position in Patient Journey")
        
        channel_positions = journey_analysis['channel_positions']
        
        if channel_positions:
            position_df = pd.DataFrame([
                {
                    'channel': channel,
                    'avg_position': data['avg_position'],
                    'first_touch_rate': data['first_touch_rate'],
                    'last_touch_rate': data['last_touch_rate'],
                    'frequency': data['frequency']
                }
                for channel, data in channel_positions.items()
            ])
            
            # Position heatmap
            fig_positions = px.scatter(
                position_df, 
                x='avg_position', 
                y='channel',
                size='frequency',
                color='first_touch_rate',
                title="Channel Position Analysis",
                labels={'avg_position': 'Average Position in Journey', 'first_touch_rate': 'First Touch Rate'},
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_positions, use_container_width=True)
            
            # Position details table
            st.subheader("Channel Position Details")
            st.dataframe(position_df.round(3))
        
    def show_forecasting_analysis(self, forecasting_results):
        """Display forecasting analysis results"""
        st.header("🔮 ROI Forecasting Analysis")
        
        # Model performance
        st.subheader("Forecasting Model Performance")
        
        performance = forecasting_results['model_performance']
        
        # Display performance for each target
        for target, perf in performance.items():
            st.write(f"**{target.replace('_', ' ').title()} Prediction:**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{perf['ensemble']['r2']:.3f}")
            with col2:
                st.metric("MAE", f"{perf['ensemble']['mae']:.2f}")
            with col3:
                st.metric("RMSE", f"{perf['ensemble']['rmse']:.2f}")
        
        # ROI forecasts
        st.subheader("ROI Forecasts")
        
        roi_forecasts = forecasting_results['roi_forecasts']
        forecast_data = roi_forecasts['forecast_data']
        
        # Actual vs Predicted comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig_roi_comparison = px.scatter(
                forecast_data, 
                x='actual_roi', 
                y='forecast_roi',
                color='channel',
                title="Actual vs Predicted ROI",
                labels={'actual_roi': 'Actual ROI', 'forecast_roi': 'Predicted ROI'}
            )
            # Add perfect prediction line
            min_roi = min(forecast_data['actual_roi'].min(), forecast_data['forecast_roi'].min())
            max_roi = max(forecast_data['actual_roi'].max(), forecast_data['forecast_roi'].max())
            fig_roi_comparison.add_trace(
                go.Scatter(x=[min_roi, max_roi], y=[min_roi, max_roi], 
                          mode='lines', name='Perfect Prediction', line=dict(dash='dash'))
            )
            st.plotly_chart(fig_roi_comparison, use_container_width=True)
        
        with col2:
            fig_conv_comparison = px.scatter(
                forecast_data, 
                x='actual_conversions', 
                y='forecast_conversions',
                color='channel',
                title="Actual vs Predicted Conversions",
                labels={'actual_conversions': 'Actual Conversions', 'forecast_conversions': 'Predicted Conversions'}
            )
            # Add perfect prediction line
            min_conv = min(forecast_data['actual_conversions'].min(), forecast_data['forecast_conversions'].min())
            max_conv = max(forecast_data['actual_conversions'].max(), forecast_data['forecast_conversions'].max())
            fig_conv_comparison.add_trace(
                go.Scatter(x=[min_conv, max_conv], y=[min_conv, max_conv], 
                          mode='lines', name='Perfect Prediction', line=dict(dash='dash'))
            )
            st.plotly_chart(fig_conv_comparison, use_container_width=True)
        
        # Budget optimization
        st.subheader("Budget Optimization Recommendations")
        
        budget_optimization = forecasting_results['budget_optimization']
        current_performance = budget_optimization['current_performance']
        optimization_scenarios = budget_optimization['optimization_scenarios']
        
        # Current performance
        st.write("**Current Channel Performance:**")
        st.dataframe(current_performance.round(2))
        
        # Optimization scenarios
        st.write("**Top Optimization Scenarios:**")
        
        for i, scenario in enumerate(optimization_scenarios, 1):
            st.write(f"**{i}. {scenario['scenario']}**")
            st.write(f"   - Budget Change: ${scenario['budget_change']:,.0f}")
            st.write(f"   - Predicted ROI Lift: {scenario['roi_lift']:.2f}%")
            st.write(f"   - Predicted Total Conversions: {scenario['predicted_total_conversions']:.0f}")
            st.write("")
        
        # Scenario analysis
        st.subheader("Scenario Analysis")
        
        scenario_analysis = forecasting_results['scenario_analysis']
        baseline = scenario_analysis['baseline']
        scenarios = scenario_analysis['scenarios']
        
        # Scenario comparison
        scenario_comparison = []
        
        for scenario_name, scenario_data in scenarios.items():
            scenario_comparison.append({
                'scenario': scenario_name,
                'description': scenario_data['description'],
                'budget_change': scenario_data['budget_change'],
                'conversions_change': scenario_data['conversions_change'],
                'roi_change': scenario_data['roi_change'],
                'total_budget': scenario_data['total_budget'],
                'total_conversions': scenario_data['total_conversions'],
                'avg_roi': scenario_data['avg_roi']
            })
        
        scenario_df = pd.DataFrame(scenario_comparison)
        
        # Scenario impact visualization
        fig_scenario = px.scatter(
            scenario_df, 
            x='budget_change', 
            y='roi_change',
            size='conversions_change',
            color='scenario',
            title="Scenario Impact Analysis",
            labels={'budget_change': 'Budget Change (%)', 'roi_change': 'ROI Change (%)', 'conversions_change': 'Conversions Change (%)'},
            hover_data=['description']
        )
        st.plotly_chart(fig_scenario, use_container_width=True)
        
        # Scenario details
        st.subheader("Scenario Details")
        st.dataframe(scenario_df.round(2))
        
    def show_database_management(self, db_manager):
        """Display database management interface"""
        st.header("🗄️ Database Management")
        
        st.subheader("Database Overview")
        
        # Database schema
        st.write("**Database Schema:**")
        st.write("- **marketing_campaigns**: Campaign performance data")
        st.write("- **patient_acquisitions**: Patient acquisition records")
        st.write("- **marketing_touchpoints**: Customer journey touchpoints")
        
        # Data exploration
        st.subheader("Data Exploration")
        
        # Sample queries
        st.write("**Sample Data Queries:**")
        
        # Campaign data
        campaign_data = db_manager.get_campaign_data()
        st.write(f"**Marketing Campaigns**: {len(campaign_data)} records")
        with st.expander("View Campaign Data Sample"):
            st.dataframe(campaign_data.head())
        
        # Patient data
        patient_data = db_manager.get_patient_data()
        st.write(f"**Patient Acquisitions**: {len(patient_data)} records")
        with st.expander("View Patient Data Sample"):
            st.dataframe(patient_data.head())
        
        # Touchpoints data
        touchpoints_data = db_manager.get_touchpoints_data()
        st.write(f"**Marketing Touchpoints**: {len(touchpoints_data)} records")
        with st.expander("View Touchpoints Data Sample"):
            st.dataframe(touchpoints_data.head())
        
        # ROI analysis
        roi_analysis = db_manager.get_roi_analysis()
        st.write(f"**ROI Analysis**: {len(roi_analysis)} records")
        with st.expander("View ROI Analysis"):
            st.dataframe(roi_analysis)
        
        # Custom query interface
        st.subheader("Custom SQL Query Interface")
        
        custom_query = st.text_area(
            "Enter your SQL query:",
            value="SELECT channel, SUM(budget) as total_budget, SUM(conversions) as total_conversions FROM marketing_campaigns GROUP BY channel ORDER BY total_budget DESC",
            height=100
        )
        
        if st.button("Execute Query"):
            try:
                result = db_manager.execute_custom_query(custom_query)
                if isinstance(result, pd.DataFrame):
                    st.write("**Query Results:**")
                    st.dataframe(result)
                else:
                    st.error(result)
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
        
        # Data quality metrics
        st.subheader("Data Quality Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Campaign Records", f"{len(campaign_data):,}")
            st.metric("Date Range", f"{campaign_data['start_date'].min()} to {campaign_data['start_date'].max()}")
        
        with col2:
            st.metric("Patient Records", f"{len(patient_data):,}")
            st.metric("Avg Patient Value", f"${patient_data['patient_value'].mean():,.0f}")
        
        with col3:
            st.metric("Touchpoint Records", f"{len(touchpoints_data):,}")
            st.metric("Avg Journey Length", f"{touchpoints_data.groupby('patient_id')['touchpoint_id'].count().mean():.1f}")
        
        # Export functionality
        st.subheader("Data Export")
        
        export_options = st.selectbox(
            "Select data to export:",
            ["Campaign Data", "Patient Data", "Touchpoints Data", "ROI Analysis"]
        )
        
        if st.button("Export to CSV"):
            if export_options == "Campaign Data":
                csv = campaign_data.to_csv(index=False)
                st.download_button(
                    label="Download Campaign Data CSV",
                    data=csv,
                    file_name="campaign_data.csv",
                    mime="text/csv"
                )
            # Add similar export options for other data types
            st.success(f"{export_options} exported successfully!") 
#!/usr/bin/env python3
"""
UPMC Healthcare Marketing Analytics Dashboard
====================================

A comprehensive analytics platform demonstrating:
- Media Mix Modeling for healthcare marketing
- Multi-Touch Attribution analysis
- ROI forecasting for marketing campaigns
- Healthcare patient acquisition analytics
- SQL database management

Author: Portfolio Project for UPMC Associate Data Scientist Role
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from database.db_manager import DatabaseManager
from models.media_mix_model import MediaMixModel
from models.attribution_model import AttributionModel
from models.forecasting_model import ROIForecastingModel
from visualization.dashboard import HealthcareMarketingDashboard

class UPMCMarketingAnalytics:
    """Main application class for UPMC Healthcare Marketing Analytics"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.media_mix_model = MediaMixModel()
        self.attribution_model = AttributionModel()
        self.forecasting_model = ROIForecastingModel()
        self.dashboard = HealthcareMarketingDashboard()
        
    def run_analysis(self):
        """Execute complete marketing analytics pipeline"""
        # Initialize database and load data
        self.db_manager.setup_database()
        
        # Load marketing campaign data
        campaign_data = self.db_manager.get_campaign_data()
        patient_data = self.db_manager.get_patient_data()
        
        # Run Media Mix Model
        mix_results = self.media_mix_model.analyze_channel_contribution(campaign_data)
        
        # Run Multi-Touch Attribution
        attribution_results = self.attribution_model.analyze_patient_journey(patient_data)
        
        # Generate ROI forecasts
        forecast_results = self.forecasting_model.predict_roi(campaign_data)
        
        return {
            'media_mix': mix_results,
            'attribution': attribution_results,
            'forecasts': forecast_results,
            'campaign_data': campaign_data,
            'patient_data': patient_data
        }

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="UPMC Healthcare Marketing Analytics",
        page_icon="🏥",
        layout="wide"
    )
    
    # Initialize analytics application
    analytics = UPMCMarketingAnalytics()
    
    # Streamlit UI
    st.title("🏥 UPMC Healthcare Marketing Analytics Dashboard")
    st.markdown("### Advanced Analytics for Healthcare Marketing ROI Optimization")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis",
        ["Overview", "Media Mix Modeling", "Attribution Analysis", "ROI Forecasting", "Database Management"]
    )
    
    # Run analysis
    with st.spinner("Loading analytics data..."):
        results = analytics.run_analysis()
    
    # Display results based on selected page
    if page == "Overview":
        analytics.dashboard.show_overview(results)
    elif page == "Media Mix Modeling":
        analytics.dashboard.show_media_mix_analysis(results['media_mix'])
    elif page == "Attribution Analysis":
        analytics.dashboard.show_attribution_analysis(results['attribution'])
    elif page == "ROI Forecasting":
        analytics.dashboard.show_forecasting_analysis(results['forecasts'])
    elif page == "Database Management":
        analytics.dashboard.show_database_management(analytics.db_manager)

if __name__ == "__main__":
    main() 
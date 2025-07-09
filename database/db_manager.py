"""
Database Manager for UPMC Healthcare Marketing Analytics
Demonstrates SQL database management skills for healthcare marketing data
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os

class DatabaseManager:
    """Manages SQLite database operations for healthcare marketing analytics"""
    
    def __init__(self, db_path="healthcare_marketing.db"):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        
    def setup_database(self):
        """Initialize database with healthcare marketing tables"""
        # Create tables using SQL
        self._create_tables()
        
        # Generate sample data
        self._generate_sample_data()
        
    def _create_tables(self):
        """Create database tables using SQL DDL"""
        create_campaigns_table = """
        CREATE TABLE IF NOT EXISTS marketing_campaigns (
            campaign_id INTEGER PRIMARY KEY,
            campaign_name TEXT NOT NULL,
            channel TEXT NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            budget REAL NOT NULL,
            impressions INTEGER,
            clicks INTEGER,
            conversions INTEGER,
            specialty TEXT,
            target_demographic TEXT
        )
        """
        
        create_patients_table = """
        CREATE TABLE IF NOT EXISTS patient_acquisitions (
            patient_id INTEGER PRIMARY KEY,
            acquisition_date DATE NOT NULL,
            specialty TEXT NOT NULL,
            patient_value REAL NOT NULL,
            acquisition_channel TEXT NOT NULL,
            touchpoint_sequence TEXT,
            demographics TEXT,
            zip_code TEXT,
            insurance_type TEXT
        )
        """
        
        create_touchpoints_table = """
        CREATE TABLE IF NOT EXISTS marketing_touchpoints (
            touchpoint_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            campaign_id INTEGER,
            touchpoint_date DATE NOT NULL,
            channel TEXT NOT NULL,
            touchpoint_type TEXT NOT NULL,
            engagement_score REAL,
            FOREIGN KEY (patient_id) REFERENCES patient_acquisitions (patient_id),
            FOREIGN KEY (campaign_id) REFERENCES marketing_campaigns (campaign_id)
        )
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(create_campaigns_table))
            conn.execute(text(create_patients_table))
            conn.execute(text(create_touchpoints_table))
            conn.commit()
            
    def _generate_sample_data(self):
        """Generate realistic healthcare marketing sample data"""
        # Check if data already exists
        existing_campaigns = self.get_campaign_data()
        if len(existing_campaigns) > 0:
            return
            
        # Generate marketing campaigns data
        campaigns_data = self._generate_campaign_data()
        patients_data = self._generate_patient_data()
        touchpoints_data = self._generate_touchpoints_data()
        
        # Insert data into database
        campaigns_data.to_sql('marketing_campaigns', self.engine, if_exists='append', index=False)
        patients_data.to_sql('patient_acquisitions', self.engine, if_exists='append', index=False)
        touchpoints_data.to_sql('marketing_touchpoints', self.engine, if_exists='append', index=False)
        
    def _generate_campaign_data(self):
        """Generate sample healthcare marketing campaign data"""
        np.random.seed(42)
        
        channels = ['Digital Display', 'Search Engine', 'Social Media', 'Email', 'TV', 'Radio', 'Print']
        specialties = ['Cardiology', 'Oncology', 'Orthopedics', 'Neurology', 'Pediatrics', 'Emergency Care']
        demographics = ['Young Adults', 'Middle Age', 'Seniors', 'Families', 'Healthcare Workers']
        
        campaigns = []
        for i in range(100):
            start_date = datetime.now() - timedelta(days=np.random.randint(30, 365))
            end_date = start_date + timedelta(days=np.random.randint(7, 90))
            
            budget = np.random.uniform(5000, 50000)
            impressions = int(budget * np.random.uniform(10, 100))
            clicks = int(impressions * np.random.uniform(0.01, 0.05))
            conversions = int(clicks * np.random.uniform(0.02, 0.15))
            
            campaigns.append({
                'campaign_id': i + 1,
                'campaign_name': f'Campaign_{i+1}',
                'channel': np.random.choice(channels),
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'budget': budget,
                'impressions': impressions,
                'clicks': clicks,
                'conversions': conversions,
                'specialty': np.random.choice(specialties),
                'target_demographic': np.random.choice(demographics)
            })
            
        return pd.DataFrame(campaigns)
        
    def _generate_patient_data(self):
        """Generate sample patient acquisition data"""
        np.random.seed(42)
        
        specialties = ['Cardiology', 'Oncology', 'Orthopedics', 'Neurology', 'Pediatrics', 'Emergency Care']
        channels = ['Digital Display', 'Search Engine', 'Social Media', 'Email', 'TV', 'Radio', 'Print']
        demographics = ['Young Adults', 'Middle Age', 'Seniors', 'Families', 'Healthcare Workers']
        insurance_types = ['Medicare', 'Medicaid', 'Private', 'Self-Pay', 'Workers Comp']
        
        patients = []
        for i in range(500):
            acquisition_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
            
            patients.append({
                'patient_id': i + 1,
                'acquisition_date': acquisition_date.strftime('%Y-%m-%d'),
                'specialty': np.random.choice(specialties),
                'patient_value': np.random.uniform(1000, 15000),
                'acquisition_channel': np.random.choice(channels),
                'touchpoint_sequence': ','.join(np.random.choice(channels, size=np.random.randint(1, 4))),
                'demographics': np.random.choice(demographics),
                'zip_code': str(np.random.randint(15000, 16000)),
                'insurance_type': np.random.choice(insurance_types)
            })
            
        return pd.DataFrame(patients)
        
    def _generate_touchpoints_data(self):
        """Generate sample touchpoint data for attribution analysis"""
        np.random.seed(42)
        
        channels = ['Digital Display', 'Search Engine', 'Social Media', 'Email', 'TV', 'Radio', 'Print']
        touchpoint_types = ['Awareness', 'Consideration', 'Decision', 'Conversion']
        
        touchpoints = []
        touchpoint_id = 1
        
        for patient_id in range(1, 501):
            num_touchpoints = np.random.randint(1, 6)
            patient_journey_start = datetime.now() - timedelta(days=np.random.randint(1, 30))
            
            for j in range(num_touchpoints):
                touchpoint_date = patient_journey_start + timedelta(days=j * np.random.randint(1, 5))
                
                touchpoints.append({
                    'touchpoint_id': touchpoint_id,
                    'patient_id': patient_id,
                    'campaign_id': np.random.randint(1, 101),
                    'touchpoint_date': touchpoint_date.strftime('%Y-%m-%d'),
                    'channel': np.random.choice(channels),
                    'touchpoint_type': touchpoint_types[min(j, 3)],
                    'engagement_score': np.random.uniform(0.1, 1.0)
                })
                touchpoint_id += 1
                
        return pd.DataFrame(touchpoints)
        
    def get_campaign_data(self):
        """Retrieve campaign data using SQL query"""
        query = """
        SELECT 
            campaign_id,
            campaign_name,
            channel,
            start_date,
            end_date,
            budget,
            impressions,
            clicks,
            conversions,
            specialty,
            target_demographic,
            CASE 
                WHEN conversions > 0 THEN budget / conversions 
                ELSE 0 
            END as cost_per_conversion
        FROM marketing_campaigns
        ORDER BY start_date DESC
        """
        
        return pd.read_sql_query(query, self.engine)
        
    def get_patient_data(self):
        """Retrieve patient acquisition data using SQL query"""
        query = """
        SELECT 
            patient_id,
            acquisition_date,
            specialty,
            patient_value,
            acquisition_channel,
            touchpoint_sequence,
            demographics,
            zip_code,
            insurance_type
        FROM patient_acquisitions
        ORDER BY acquisition_date DESC
        """
        
        return pd.read_sql_query(query, self.engine)
        
    def get_touchpoints_data(self):
        """Retrieve touchpoint data for attribution analysis"""
        query = """
        SELECT 
            t.touchpoint_id,
            t.patient_id,
            t.campaign_id,
            t.touchpoint_date,
            t.channel,
            t.touchpoint_type,
            t.engagement_score,
            c.budget,
            c.specialty,
            p.patient_value
        FROM marketing_touchpoints t
        LEFT JOIN marketing_campaigns c ON t.campaign_id = c.campaign_id
        LEFT JOIN patient_acquisitions p ON t.patient_id = p.patient_id
        ORDER BY t.patient_id, t.touchpoint_date
        """
        
        return pd.read_sql_query(query, self.engine)
        
    def get_roi_analysis(self):
        """Calculate ROI metrics using SQL"""
        query = """
        SELECT 
            c.channel,
            c.specialty,
            SUM(c.budget) as total_budget,
            SUM(c.conversions) as total_conversions,
            AVG(p.patient_value) as avg_patient_value,
            SUM(c.conversions * p.patient_value) as total_revenue,
            (SUM(c.conversions * p.patient_value) - SUM(c.budget)) / SUM(c.budget) * 100 as roi_percentage
        FROM marketing_campaigns c
        LEFT JOIN patient_acquisitions p ON c.channel = p.acquisition_channel
        GROUP BY c.channel, c.specialty
        HAVING total_conversions > 0
        ORDER BY roi_percentage DESC
        """
        
        return pd.read_sql_query(query, self.engine)
        
    def execute_custom_query(self, query):
        """Execute custom SQL query"""
        try:
            return pd.read_sql_query(query, self.engine)
        except Exception as e:
            return f"Error executing query: {str(e)}" 
"""
ROI Forecasting Model for UPMC Healthcare Marketing Analytics
Demonstrates predictive modeling for marketing ROI and budget optimization
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ROIForecastingModel:
    """
    ROI Forecasting Model for healthcare marketing budget optimization
    Uses ensemble methods to predict campaign performance and ROI
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.forecast_results = None
        
    def predict_roi(self, campaign_data):
        """
        Predict ROI for marketing campaigns and generate forecasts
        
        Args:
            campaign_data (pd.DataFrame): Historical campaign data
            
        Returns:
            dict: Forecasting results and recommendations
        """
        # Prepare data for modeling
        model_data = self._prepare_forecasting_data(campaign_data)
        
        # Build ROI prediction models
        model_results = self._build_forecasting_models(model_data)
        
        # Generate ROI forecasts
        roi_forecasts = self._generate_roi_forecasts(model_data, model_results)
        
        # Optimize budget allocation
        budget_optimization = self._optimize_budget_allocation(
            model_data, model_results, campaign_data
        )
        
        # Perform scenario analysis
        scenario_analysis = self._perform_scenario_analysis(
            model_data, model_results, campaign_data
        )
        
        return {
            'roi_forecasts': roi_forecasts,
            'budget_optimization': budget_optimization,
            'scenario_analysis': scenario_analysis,
            'model_performance': model_results['performance'],
            'feature_importance': model_results['feature_importance']
        }
        
    def _prepare_forecasting_data(self, campaign_data):
        """Prepare data for forecasting models"""
        # Create time-based features
        campaign_data['start_date'] = pd.to_datetime(campaign_data['start_date'])
        campaign_data['end_date'] = pd.to_datetime(campaign_data['end_date'])
        
        # Calculate campaign duration
        campaign_data['campaign_duration'] = (
            campaign_data['end_date'] - campaign_data['start_date']
        ).dt.days
        
        # Create seasonal features
        campaign_data['month'] = campaign_data['start_date'].dt.month
        campaign_data['quarter'] = campaign_data['start_date'].dt.quarter
        campaign_data['day_of_week'] = campaign_data['start_date'].dt.dayofweek
        
        # Calculate performance metrics
        campaign_data['ctr'] = campaign_data['clicks'] / campaign_data['impressions']
        campaign_data['conversion_rate'] = campaign_data['conversions'] / campaign_data['clicks']
        campaign_data['cost_per_click'] = campaign_data['budget'] / campaign_data['clicks']
        campaign_data['cost_per_conversion'] = campaign_data['budget'] / campaign_data['conversions']
        
        # Calculate ROI (assuming average patient value)
        avg_patient_value = 5000  # Assumed average patient value
        campaign_data['revenue'] = campaign_data['conversions'] * avg_patient_value
        campaign_data['roi'] = (campaign_data['revenue'] - campaign_data['budget']) / campaign_data['budget']
        
        # Create lag features
        campaign_data = campaign_data.sort_values('start_date')
        campaign_data['prev_month_conversions'] = campaign_data.groupby('channel')['conversions'].shift(1)
        campaign_data['prev_month_roi'] = campaign_data.groupby('channel')['roi'].shift(1)
        
        # Create channel dummy variables
        channel_dummies = pd.get_dummies(campaign_data['channel'], prefix='channel')
        specialty_dummies = pd.get_dummies(campaign_data['specialty'], prefix='specialty')
        demo_dummies = pd.get_dummies(campaign_data['target_demographic'], prefix='demo')
        
        # Combine features
        feature_columns = [
            'budget', 'impressions', 'clicks', 'campaign_duration',
            'month', 'quarter', 'day_of_week', 'ctr', 'conversion_rate',
            'cost_per_click', 'prev_month_conversions', 'prev_month_roi'
        ]
        
        features = pd.concat([
            campaign_data[feature_columns],
            channel_dummies,
            specialty_dummies,
            demo_dummies
        ], axis=1).fillna(0)
        
        # Target variables
        targets = {
            'conversions': campaign_data['conversions'],
            'roi': campaign_data['roi'],
            'cost_per_conversion': campaign_data['cost_per_conversion']
        }
        
        return {
            'features': features,
            'targets': targets,
            'campaign_data': campaign_data
        }
        
    def _build_forecasting_models(self, model_data):
        """Build ensemble forecasting models"""
        X = model_data['features']
        performance_results = {}
        feature_importance_results = {}
        
        # Build models for each target variable
        for target_name, y in model_data['targets'].items():
            # Remove rows with missing target values
            valid_idx = ~y.isna() & ~np.isinf(y)
            X_clean = X.loc[valid_idx]
            y_clean = y.loc[valid_idx]
            
            if len(X_clean) == 0:
                continue
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.3, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Build ensemble model
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'lr': LinearRegression()
            }
            
            model_predictions = {}
            model_scores = {}
            
            # Train individual models
            for model_name, model in models.items():
                if model_name == 'lr':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                model_predictions[model_name] = y_pred
                model_scores[model_name] = {
                    'r2': r2_score(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                }
            
            # Create ensemble prediction (weighted average)
            ensemble_prediction = (
                0.4 * model_predictions['rf'] +
                0.4 * model_predictions['gb'] +
                0.2 * model_predictions['lr']
            )
            
            # Evaluate ensemble
            ensemble_score = {
                'r2': r2_score(y_test, ensemble_prediction),
                'mae': mean_absolute_error(y_test, ensemble_prediction),
                'rmse': np.sqrt(mean_squared_error(y_test, ensemble_prediction))
            }
            
            # Store results
            performance_results[target_name] = {
                'individual_models': model_scores,
                'ensemble': ensemble_score
            }
            
            # Feature importance (from Random Forest)
            rf_model = models['rf']
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance_results[target_name] = feature_importance
            
            # Store trained models
            self.models[target_name] = {
                'rf': rf_model,
                'gb': models['gb'],
                'lr': models['lr']
            }
            self.scalers[target_name] = scaler
            
        return {
            'performance': performance_results,
            'feature_importance': feature_importance_results
        }
        
    def _generate_roi_forecasts(self, model_data, model_results):
        """Generate ROI forecasts for different scenarios"""
        X = model_data['features']
        campaign_data = model_data['campaign_data']
        
        forecasts = {}
        
        # Generate forecasts for each target
        for target_name in ['conversions', 'roi', 'cost_per_conversion']:
            if target_name in self.models:
                # Use ensemble prediction
                rf_pred = self.models[target_name]['rf'].predict(X)
                gb_pred = self.models[target_name]['gb'].predict(X)
                
                X_scaled = self.scalers[target_name].transform(X)
                lr_pred = self.models[target_name]['lr'].predict(X_scaled)
                
                # Ensemble prediction
                ensemble_pred = 0.4 * rf_pred + 0.4 * gb_pred + 0.2 * lr_pred
                
                forecasts[target_name] = ensemble_pred
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'campaign_id': campaign_data['campaign_id'],
            'channel': campaign_data['channel'],
            'specialty': campaign_data['specialty'],
            'actual_conversions': campaign_data['conversions'],
            'forecast_conversions': forecasts.get('conversions', 0),
            'actual_roi': campaign_data['roi'],
            'forecast_roi': forecasts.get('roi', 0),
            'actual_cost_per_conversion': campaign_data['cost_per_conversion'],
            'forecast_cost_per_conversion': forecasts.get('cost_per_conversion', 0),
            'budget': campaign_data['budget']
        })
        
        # Calculate forecast accuracy
        forecast_accuracy = {}
        for target in ['conversions', 'roi', 'cost_per_conversion']:
            if f'forecast_{target}' in forecast_df.columns:
                actual = forecast_df[f'actual_{target}']
                predicted = forecast_df[f'forecast_{target}']
                
                # Remove infinite values
                valid_idx = ~np.isinf(actual) & ~np.isinf(predicted)
                actual_clean = actual[valid_idx]
                predicted_clean = predicted[valid_idx]
                
                if len(actual_clean) > 0:
                    forecast_accuracy[target] = {
                        'mape': np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100,
                        'mae': mean_absolute_error(actual_clean, predicted_clean),
                        'r2': r2_score(actual_clean, predicted_clean)
                    }
        
        return {
            'forecast_data': forecast_df,
            'forecast_accuracy': forecast_accuracy
        }
        
    def _optimize_budget_allocation(self, model_data, model_results, campaign_data):
        """Optimize budget allocation across channels"""
        # Current performance by channel
        current_performance = campaign_data.groupby('channel').agg({
            'budget': 'sum',
            'conversions': 'sum',
            'roi': 'mean'
        }).reset_index()
        
        current_performance['current_cost_per_conversion'] = (
            current_performance['budget'] / current_performance['conversions']
        )
        
        # Simulate budget reallocation scenarios
        total_budget = current_performance['budget'].sum()
        optimization_scenarios = []
        
        # Generate optimization scenarios
        for channel in current_performance['channel'].unique():
            # Increase budget by 20%
            scenario_budget = current_performance.copy()
            scenario_budget.loc[scenario_budget['channel'] == channel, 'budget'] *= 1.2
            
            # Decrease other channels proportionally
            other_channels = scenario_budget['channel'] != channel
            reduction_factor = (total_budget - scenario_budget.loc[scenario_budget['channel'] == channel, 'budget'].sum()) / scenario_budget.loc[other_channels, 'budget'].sum()
            scenario_budget.loc[other_channels, 'budget'] *= reduction_factor
            
            # Predict performance with new budget
            predicted_roi = self._predict_scenario_performance(
                scenario_budget, model_data, current_performance
            )
            
            optimization_scenarios.append({
                'scenario': f'Increase {channel} budget by 20%',
                'channel': channel,
                'budget_change': scenario_budget.loc[scenario_budget['channel'] == channel, 'budget'].iloc[0] - current_performance.loc[current_performance['channel'] == channel, 'budget'].iloc[0],
                'predicted_total_roi': predicted_roi['total_roi'],
                'predicted_total_conversions': predicted_roi['total_conversions'],
                'roi_lift': predicted_roi['roi_lift']
            })
        
        # Sort by predicted ROI improvement
        optimization_scenarios = sorted(optimization_scenarios, key=lambda x: x['roi_lift'], reverse=True)
        
        return {
            'current_performance': current_performance,
            'optimization_scenarios': optimization_scenarios[:5],  # Top 5 scenarios
            'total_budget': total_budget
        }
        
    def _predict_scenario_performance(self, scenario_budget, model_data, current_performance):
        """Predict performance for a budget scenario"""
        # Simplified prediction based on current performance and elasticity
        total_conversions = 0
        total_roi = 0
        
        for _, row in scenario_budget.iterrows():
            channel = row['channel']
            new_budget = row['budget']
            
            # Get current performance for this channel
            current_channel_data = current_performance[current_performance['channel'] == channel]
            if len(current_channel_data) > 0:
                current_budget = current_channel_data.iloc[0]['budget']
                current_conversions = current_channel_data.iloc[0]['conversions']
                current_roi = current_channel_data.iloc[0]['roi']
                
                # Apply diminishing returns (elasticity of 0.7)
                budget_ratio = new_budget / current_budget if current_budget > 0 else 1
                elasticity = 0.7
                
                predicted_conversions = current_conversions * (budget_ratio ** elasticity)
                predicted_roi = current_roi * (budget_ratio ** (elasticity - 1))
                
                total_conversions += predicted_conversions
                total_roi += predicted_roi * new_budget
        
        # Calculate overall metrics
        total_budget = scenario_budget['budget'].sum()
        avg_roi = total_roi / total_budget if total_budget > 0 else 0
        
        # Calculate improvement
        current_total_roi = (current_performance['roi'] * current_performance['budget']).sum() / current_performance['budget'].sum()
        roi_lift = (avg_roi - current_total_roi) / current_total_roi * 100 if current_total_roi > 0 else 0
        
        return {
            'total_conversions': total_conversions,
            'total_roi': avg_roi,
            'roi_lift': roi_lift
        }
        
    def _perform_scenario_analysis(self, model_data, model_results, campaign_data):
        """Perform scenario analysis for different market conditions"""
        scenarios = {
            'Economic Downturn': {
                'budget_multiplier': 0.8,
                'conversion_rate_multiplier': 0.9,
                'description': '20% budget cut, 10% lower conversion rates'
            },
            'Market Expansion': {
                'budget_multiplier': 1.3,
                'conversion_rate_multiplier': 1.1,
                'description': '30% budget increase, 10% higher conversion rates'
            },
            'Competitive Pressure': {
                'budget_multiplier': 1.0,
                'conversion_rate_multiplier': 0.85,
                'description': 'Same budget, 15% lower conversion rates'
            },
            'Seasonal Peak': {
                'budget_multiplier': 1.2,
                'conversion_rate_multiplier': 1.15,
                'description': '20% budget increase, 15% higher conversion rates'
            }
        }
        
        scenario_results = {}
        
        # Current baseline
        current_total_budget = campaign_data['budget'].sum()
        current_total_conversions = campaign_data['conversions'].sum()
        current_avg_roi = campaign_data['roi'].mean()
        
        for scenario_name, scenario_params in scenarios.items():
            # Apply scenario parameters
            scenario_budget = current_total_budget * scenario_params['budget_multiplier']
            scenario_conversions = current_total_conversions * scenario_params['conversion_rate_multiplier']
            
            # Calculate scenario ROI
            avg_patient_value = 5000
            scenario_revenue = scenario_conversions * avg_patient_value
            scenario_roi = (scenario_revenue - scenario_budget) / scenario_budget
            
            # Channel-level analysis
            channel_analysis = campaign_data.groupby('channel').agg({
                'budget': 'sum',
                'conversions': 'sum',
                'roi': 'mean'
            }).reset_index()
            
            channel_analysis['scenario_budget'] = channel_analysis['budget'] * scenario_params['budget_multiplier']
            channel_analysis['scenario_conversions'] = channel_analysis['conversions'] * scenario_params['conversion_rate_multiplier']
            channel_analysis['scenario_roi'] = (
                (channel_analysis['scenario_conversions'] * avg_patient_value - channel_analysis['scenario_budget']) / 
                channel_analysis['scenario_budget']
            )
            
            scenario_results[scenario_name] = {
                'description': scenario_params['description'],
                'total_budget': scenario_budget,
                'total_conversions': scenario_conversions,
                'avg_roi': scenario_roi,
                'budget_change': (scenario_budget - current_total_budget) / current_total_budget * 100,
                'conversions_change': (scenario_conversions - current_total_conversions) / current_total_conversions * 100,
                'roi_change': (scenario_roi - current_avg_roi) / current_avg_roi * 100,
                'channel_analysis': channel_analysis
            }
        
        return {
            'baseline': {
                'total_budget': current_total_budget,
                'total_conversions': current_total_conversions,
                'avg_roi': current_avg_roi
            },
            'scenarios': scenario_results
        } 
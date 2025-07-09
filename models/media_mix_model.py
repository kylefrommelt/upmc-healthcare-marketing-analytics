"""
Media Mix Model for UPMC Healthcare Marketing Analytics
Demonstrates econometric modeling for marketing channel effectiveness analysis
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MediaMixModel:
    """
    Media Mix Model for analyzing marketing channel contributions
    Implements econometric modeling techniques for healthcare marketing ROI analysis
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.model_performance = None
        
    def analyze_channel_contribution(self, campaign_data):
        """
        Analyze the contribution of each marketing channel to patient acquisition
        
        Args:
            campaign_data (pd.DataFrame): Campaign performance data
            
        Returns:
            dict: Analysis results including channel contributions and model performance
        """
        # Prepare data for modeling
        model_data = self._prepare_modeling_data(campaign_data)
        
        # Build media mix model
        model_results = self._build_media_mix_model(model_data)
        
        # Calculate channel contributions
        channel_contributions = self._calculate_channel_contributions(model_data, model_results)
        
        # Perform elasticity analysis
        elasticity_analysis = self._calculate_elasticity(model_data, model_results)
        
        # Generate optimization recommendations
        optimization_recommendations = self._generate_optimization_recommendations(
            channel_contributions, elasticity_analysis, campaign_data
        )
        
        return {
            'channel_contributions': channel_contributions,
            'elasticity_analysis': elasticity_analysis,
            'model_performance': model_results['performance'],
            'optimization_recommendations': optimization_recommendations,
            'model_data': model_data
        }
        
    def _prepare_modeling_data(self, campaign_data):
        """Prepare data for media mix modeling"""
        # Aggregate data by channel and time period
        campaign_data['start_date'] = pd.to_datetime(campaign_data['start_date'])
        campaign_data['month'] = campaign_data['start_date'].dt.to_period('M')
        
        # Create channel-level aggregation
        channel_monthly = campaign_data.groupby(['month', 'channel']).agg({
            'budget': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum'
        }).reset_index()
        
        # Pivot to create feature matrix
        features = []
        for channel in campaign_data['channel'].unique():
            channel_data = channel_monthly[channel_monthly['channel'] == channel]
            channel_features = channel_data.pivot_table(
                index='month', 
                values=['budget', 'impressions'], 
                aggfunc='sum'
            ).fillna(0)
            
            # Add channel prefix to column names
            channel_features.columns = [f"{channel}_{col}" for col in channel_features.columns]
            features.append(channel_features)
        
        # Combine all channel features
        feature_matrix = pd.concat(features, axis=1).fillna(0)
        
        # Create target variable (total conversions by month)
        target = campaign_data.groupby('month')['conversions'].sum()
        
        # Align indices
        common_index = feature_matrix.index.intersection(target.index)
        feature_matrix = feature_matrix.loc[common_index]
        target = target.loc[common_index]
        
        # Add adstock transformation (carryover effects)
        feature_matrix = self._apply_adstock_transformation(feature_matrix)
        
        return {
            'features': feature_matrix,
            'target': target,
            'channel_data': channel_monthly
        }
        
    def _apply_adstock_transformation(self, feature_matrix, adstock_rate=0.5):
        """Apply adstock transformation to account for carryover effects"""
        transformed_features = feature_matrix.copy()
        
        for col in feature_matrix.columns:
            if 'budget' in col:
                # Apply adstock transformation
                adstocked_values = []
                for i in range(len(feature_matrix)):
                    if i == 0:
                        adstocked_values.append(feature_matrix[col].iloc[i])
                    else:
                        adstocked_value = (feature_matrix[col].iloc[i] + 
                                         adstock_rate * adstocked_values[i-1])
                        adstocked_values.append(adstocked_value)
                
                transformed_features[f"{col}_adstock"] = adstocked_values
        
        return transformed_features
        
    def _build_media_mix_model(self, model_data):
        """Build the media mix model using multiple regression techniques"""
        X = model_data['features']
        y = model_data['target']
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fit Ridge regression (handles multicollinearity)
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate performance metrics
        performance = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return {
            'model': self.model,
            'performance': performance,
            'feature_importance': feature_importance,
            'predictions': {'train': y_pred_train, 'test': y_pred_test},
            'actual': {'train': y_train, 'test': y_test}
        }
        
    def _calculate_channel_contributions(self, model_data, model_results):
        """Calculate the contribution of each channel to total conversions"""
        X = model_data['features']
        X_scaled = self.scaler.transform(X)
        
        # Calculate contribution for each channel
        contributions = {}
        
        for channel in ['Digital Display', 'Search Engine', 'Social Media', 'Email', 'TV', 'Radio', 'Print']:
            # Find relevant features for this channel
            channel_features = [col for col in X.columns if channel.replace(' ', '_') in col or channel in col]
            
            if channel_features:
                # Calculate contribution
                channel_contribution = 0
                for feature in channel_features:
                    if feature in X.columns:
                        feature_idx = list(X.columns).index(feature)
                        contribution = (X_scaled[:, feature_idx] * 
                                     model_results['feature_importance'].iloc[feature_idx]['coefficient'])
                        channel_contribution += np.sum(contribution)
                
                contributions[channel] = {
                    'total_contribution': channel_contribution,
                    'contribution_percentage': 0,  # Will be calculated below
                    'features': channel_features
                }
        
        # Calculate percentages
        total_contribution = sum([c['total_contribution'] for c in contributions.values()])
        
        if total_contribution > 0:
            for channel in contributions:
                contributions[channel]['contribution_percentage'] = (
                    contributions[channel]['total_contribution'] / total_contribution * 100
                )
        
        return contributions
        
    def _calculate_elasticity(self, model_data, model_results):
        """Calculate price elasticity for each marketing channel"""
        X = model_data['features']
        elasticity_results = {}
        
        for channel in ['Digital Display', 'Search Engine', 'Social Media', 'Email', 'TV', 'Radio', 'Print']:
            budget_col = f"{channel}_budget"
            
            if budget_col in X.columns:
                # Calculate elasticity: (% change in conversions) / (% change in budget)
                budget_mean = X[budget_col].mean()
                
                if budget_mean > 0:
                    # Find coefficient for this channel
                    feature_idx = list(X.columns).index(budget_col)
                    coefficient = model_results['feature_importance'].iloc[feature_idx]['coefficient']
                    
                    # Calculate elasticity
                    elasticity = coefficient * (budget_mean / model_data['target'].mean())
                    
                    elasticity_results[channel] = {
                        'elasticity': elasticity,
                        'interpretation': self._interpret_elasticity(elasticity)
                    }
        
        return elasticity_results
        
    def _interpret_elasticity(self, elasticity):
        """Interpret elasticity values"""
        if elasticity > 1:
            return "Elastic - High responsiveness to budget changes"
        elif elasticity > 0.5:
            return "Moderately elastic - Good responsiveness"
        elif elasticity > 0:
            return "Inelastic - Low responsiveness to budget changes"
        else:
            return "Negative elasticity - Decreasing returns"
            
    def _generate_optimization_recommendations(self, channel_contributions, elasticity_analysis, campaign_data):
        """Generate budget optimization recommendations"""
        recommendations = []
        
        # Calculate current performance metrics
        current_performance = campaign_data.groupby('channel').agg({
            'budget': 'sum',
            'conversions': 'sum'
        }).reset_index()
        
        current_performance['cost_per_conversion'] = (
            current_performance['budget'] / current_performance['conversions']
        )
        
        # Sort by efficiency
        current_performance = current_performance.sort_values('cost_per_conversion')
        
        # Generate recommendations
        for _, row in current_performance.iterrows():
            channel = row['channel']
            
            # Get elasticity info
            elasticity_info = elasticity_analysis.get(channel, {})
            elasticity = elasticity_info.get('elasticity', 0)
            
            # Get contribution info
            contribution_info = channel_contributions.get(channel, {})
            contribution_pct = contribution_info.get('contribution_percentage', 0)
            
            # Generate recommendation
            if elasticity > 0.8 and contribution_pct > 15:
                recommendation = f"INCREASE budget for {channel} - High elasticity and significant contribution"
            elif elasticity < 0.3 and contribution_pct < 5:
                recommendation = f"DECREASE budget for {channel} - Low elasticity and minimal contribution"
            elif row['cost_per_conversion'] < current_performance['cost_per_conversion'].median():
                recommendation = f"MAINTAIN or INCREASE budget for {channel} - Cost-efficient channel"
            else:
                recommendation = f"OPTIMIZE {channel} campaigns - Review targeting and creative"
                
            recommendations.append({
                'channel': channel,
                'recommendation': recommendation,
                'current_budget': row['budget'],
                'cost_per_conversion': row['cost_per_conversion'],
                'elasticity': elasticity,
                'contribution_percentage': contribution_pct
            })
        
        return recommendations 
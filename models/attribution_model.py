"""
Multi-Touch Attribution Model for UPMC Healthcare Marketing Analytics
Demonstrates advanced attribution modeling for healthcare patient journey analysis
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class AttributionModel:
    """
    Multi-Touch Attribution Model for healthcare marketing
    Analyzes patient journey touchpoints to determine channel effectiveness
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.attribution_results = None
        
    def analyze_patient_journey(self, patient_data):
        """
        Analyze patient journey touchpoints and attribute conversions
        
        Args:
            patient_data (pd.DataFrame): Patient acquisition data
            
        Returns:
            dict: Attribution analysis results
        """
        # Prepare touchpoint data
        touchpoint_data = self._prepare_touchpoint_data(patient_data)
        
        # Build attribution model
        model_results = self._build_attribution_model(touchpoint_data)
        
        # Calculate attribution weights
        attribution_weights = self._calculate_attribution_weights(touchpoint_data, model_results)
        
        # Perform journey analysis
        journey_analysis = self._analyze_patient_journeys(touchpoint_data, attribution_weights)
        
        # Generate channel attribution report
        channel_attribution = self._generate_channel_attribution_report(
            touchpoint_data, attribution_weights, patient_data
        )
        
        return {
            'attribution_weights': attribution_weights,
            'journey_analysis': journey_analysis,
            'channel_attribution': channel_attribution,
            'model_performance': model_results['performance'],
            'touchpoint_data': touchpoint_data
        }
        
    def _prepare_touchpoint_data(self, patient_data):
        """Prepare touchpoint data for attribution modeling"""
        # Simulate touchpoint data based on patient acquisition data
        touchpoints = []
        
        for _, patient in patient_data.iterrows():
            # Parse touchpoint sequence
            touchpoint_sequence = patient['touchpoint_sequence'].split(',')
            
            # Create touchpoint records
            for i, channel in enumerate(touchpoint_sequence):
                touchpoint_date = pd.to_datetime(patient['acquisition_date']) - pd.Timedelta(days=(len(touchpoint_sequence) - i - 1) * 2)
                
                touchpoints.append({
                    'patient_id': patient['patient_id'],
                    'touchpoint_order': i + 1,
                    'channel': channel.strip(),
                    'touchpoint_date': touchpoint_date,
                    'days_to_conversion': (len(touchpoint_sequence) - i - 1) * 2,
                    'patient_value': patient['patient_value'],
                    'specialty': patient['specialty'],
                    'demographics': patient['demographics'],
                    'converted': 1  # All patients in this dataset converted
                })
        
        touchpoint_df = pd.DataFrame(touchpoints)
        
        # Add non-converting touchpoints for model training
        non_converting_touchpoints = self._generate_non_converting_touchpoints(touchpoint_df)
        
        # Combine converting and non-converting touchpoints
        all_touchpoints = pd.concat([touchpoint_df, non_converting_touchpoints], ignore_index=True)
        
        return all_touchpoints
        
    def _generate_non_converting_touchpoints(self, converting_touchpoints):
        """Generate non-converting touchpoints for model balance"""
        non_converting = []
        
        channels = converting_touchpoints['channel'].unique()
        specialties = converting_touchpoints['specialty'].unique()
        demographics = converting_touchpoints['demographics'].unique()
        
        # Generate non-converting touchpoints (simulate users who didn't convert)
        for i in range(len(converting_touchpoints) // 2):  # 2:1 ratio converting:non-converting
            patient_id = f"non_convert_{i}"
            num_touchpoints = np.random.randint(1, 4)
            
            for j in range(num_touchpoints):
                non_converting.append({
                    'patient_id': patient_id,
                    'touchpoint_order': j + 1,
                    'channel': np.random.choice(channels),
                    'touchpoint_date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 30)),
                    'days_to_conversion': np.random.randint(1, 30),
                    'patient_value': 0,  # No value for non-converting patients
                    'specialty': np.random.choice(specialties),
                    'demographics': np.random.choice(demographics),
                    'converted': 0
                })
        
        return pd.DataFrame(non_converting)
        
    def _build_attribution_model(self, touchpoint_data):
        """Build logistic regression model for attribution analysis"""
        # Prepare features
        feature_data = self._prepare_model_features(touchpoint_data)
        
        # Split data
        X = feature_data.drop(['converted', 'patient_id', 'patient_value'], axis=1)
        y = feature_data['converted']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train logistic regression model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        y_pred_proba_train = self.model.predict_proba(X_train_scaled)[:, 1]
        y_pred_proba_test = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate performance metrics
        performance = {
            'train_auc': roc_auc_score(y_train, y_pred_proba_train),
            'test_auc': roc_auc_score(y_test, y_pred_proba_test),
            'train_accuracy': (y_pred_train == y_train).mean(),
            'test_accuracy': (y_pred_test == y_test).mean()
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        return {
            'model': self.model,
            'performance': performance,
            'feature_importance': feature_importance,
            'feature_data': feature_data
        }
        
    def _prepare_model_features(self, touchpoint_data):
        """Prepare features for attribution model"""
        # Create channel dummy variables
        channel_dummies = pd.get_dummies(touchpoint_data['channel'], prefix='channel')
        
        # Create specialty dummy variables
        specialty_dummies = pd.get_dummies(touchpoint_data['specialty'], prefix='specialty')
        
        # Create demographic dummy variables
        demographics_dummies = pd.get_dummies(touchpoint_data['demographics'], prefix='demographics')
        
        # Create position-based features
        touchpoint_data['is_first_touch'] = (touchpoint_data['touchpoint_order'] == 1).astype(int)
        touchpoint_data['is_last_touch'] = touchpoint_data.groupby('patient_id')['touchpoint_order'].transform('max') == touchpoint_data['touchpoint_order']
        touchpoint_data['is_last_touch'] = touchpoint_data['is_last_touch'].astype(int)
        
        # Time-based features
        touchpoint_data['days_to_conversion_log'] = np.log1p(touchpoint_data['days_to_conversion'])
        touchpoint_data['recency_score'] = 1 / (1 + touchpoint_data['days_to_conversion'])
        
        # Combine all features
        feature_columns = [
            'touchpoint_order', 'days_to_conversion', 'days_to_conversion_log',
            'recency_score', 'is_first_touch', 'is_last_touch', 'converted',
            'patient_id', 'patient_value'
        ]
        
        features = pd.concat([
            touchpoint_data[feature_columns],
            channel_dummies,
            specialty_dummies,
            demographics_dummies
        ], axis=1)
        
        return features
        
    def _calculate_attribution_weights(self, touchpoint_data, model_results):
        """Calculate attribution weights for each touchpoint"""
        feature_data = model_results['feature_data']
        
        # Prepare features for prediction
        X = feature_data.drop(['converted', 'patient_id', 'patient_value'], axis=1)
        X_scaled = self.scaler.transform(X)
        
        # Get conversion probabilities
        conversion_probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate attribution weights
        attribution_weights = []
        
        for patient_id in touchpoint_data['patient_id'].unique():
            if patient_id.startswith('non_convert'):
                continue
                
            patient_touchpoints = touchpoint_data[touchpoint_data['patient_id'] == patient_id]
            patient_features = feature_data[feature_data['patient_id'] == patient_id]
            
            if len(patient_touchpoints) > 0:
                # Get probabilities for this patient's touchpoints
                patient_indices = patient_features.index
                patient_probabilities = conversion_probabilities[patient_indices]
                
                # Calculate attribution weights using Shapley-based approach
                total_probability = np.sum(patient_probabilities)
                
                for i, (_, touchpoint) in enumerate(patient_touchpoints.iterrows()):
                    if total_probability > 0:
                        shapley_weight = patient_probabilities[i] / total_probability
                        
                        # Apply position-based adjustments
                        if touchpoint['touchpoint_order'] == 1:  # First touch
                            position_weight = 0.4
                        elif touchpoint['touchpoint_order'] == len(patient_touchpoints):  # Last touch
                            position_weight = 0.4
                        else:  # Middle touches
                            position_weight = 0.2 / max(1, len(patient_touchpoints) - 2)
                        
                        # Combine Shapley and position weights
                        final_weight = 0.6 * shapley_weight + 0.4 * position_weight
                        
                        attribution_weights.append({
                            'patient_id': patient_id,
                            'touchpoint_order': touchpoint['touchpoint_order'],
                            'channel': touchpoint['channel'],
                            'attribution_weight': final_weight,
                            'patient_value': touchpoint['patient_value'],
                            'attributed_value': final_weight * touchpoint['patient_value']
                        })
        
        return pd.DataFrame(attribution_weights)
        
    def _analyze_patient_journeys(self, touchpoint_data, attribution_weights):
        """Analyze patient journey patterns"""
        converting_patients = touchpoint_data[touchpoint_data['converted'] == 1]
        
        # Journey length analysis
        journey_lengths = converting_patients.groupby('patient_id')['touchpoint_order'].max()
        
        # Common journey patterns
        journey_patterns = converting_patients.groupby('patient_id')['channel'].apply(
            lambda x: ' → '.join(x.values)
        ).value_counts().head(10)
        
        # Channel position analysis
        channel_positions = {}
        for channel in converting_patients['channel'].unique():
            channel_data = converting_patients[converting_patients['channel'] == channel]
            
            channel_positions[channel] = {
                'avg_position': channel_data['touchpoint_order'].mean(),
                'first_touch_rate': (channel_data['touchpoint_order'] == 1).mean(),
                'last_touch_rate': channel_data.groupby('patient_id')['touchpoint_order'].transform('max') == channel_data['touchpoint_order'],
                'frequency': len(channel_data)
            }
            
            # Calculate last touch rate properly
            last_touch_count = 0
            for patient_id in channel_data['patient_id'].unique():
                patient_touchpoints = converting_patients[converting_patients['patient_id'] == patient_id]
                max_order = patient_touchpoints['touchpoint_order'].max()
                if any((patient_touchpoints['channel'] == channel) & (patient_touchpoints['touchpoint_order'] == max_order)):
                    last_touch_count += 1
            
            channel_positions[channel]['last_touch_rate'] = last_touch_count / len(channel_data['patient_id'].unique())
        
        # Time between touchpoints
        time_between_touchpoints = converting_patients.groupby('patient_id').apply(
            lambda x: x.sort_values('touchpoint_order')['days_to_conversion'].diff().abs().mean()
        ).mean()
        
        return {
            'avg_journey_length': journey_lengths.mean(),
            'journey_length_distribution': journey_lengths.value_counts().to_dict(),
            'common_journey_patterns': journey_patterns.to_dict(),
            'channel_positions': channel_positions,
            'avg_time_between_touchpoints': time_between_touchpoints
        }
        
    def _generate_channel_attribution_report(self, touchpoint_data, attribution_weights, patient_data):
        """Generate comprehensive channel attribution report"""
        # Channel-level attribution
        channel_attribution = attribution_weights.groupby('channel').agg({
            'attribution_weight': 'sum',
            'attributed_value': 'sum',
            'patient_id': 'nunique'
        }).reset_index()
        
        channel_attribution.columns = ['channel', 'total_attribution_weight', 'total_attributed_value', 'unique_patients']
        
        # Calculate attribution percentages
        total_attributed_value = channel_attribution['total_attributed_value'].sum()
        channel_attribution['attribution_percentage'] = (
            channel_attribution['total_attributed_value'] / total_attributed_value * 100
        )
        
        # Add last-touch attribution for comparison
        last_touch_attribution = patient_data.groupby('acquisition_channel').agg({
            'patient_value': 'sum',
            'patient_id': 'count'
        }).reset_index()
        
        last_touch_attribution.columns = ['channel', 'last_touch_value', 'last_touch_conversions']
        
        # Merge attribution models
        attribution_comparison = pd.merge(
            channel_attribution, 
            last_touch_attribution, 
            on='channel', 
            how='outer'
        ).fillna(0)
        
        # Calculate attribution lift (multi-touch vs last-touch)
        attribution_comparison['attribution_lift'] = (
            (attribution_comparison['total_attributed_value'] - attribution_comparison['last_touch_value']) / 
            attribution_comparison['last_touch_value'] * 100
        )
        
        # Sort by attributed value
        attribution_comparison = attribution_comparison.sort_values('total_attributed_value', ascending=False)
        
        return {
            'channel_attribution': channel_attribution,
            'attribution_comparison': attribution_comparison,
            'total_attributed_value': total_attributed_value
        } 
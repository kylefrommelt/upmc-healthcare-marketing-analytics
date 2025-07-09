# UPMC Healthcare Marketing Analytics Dashboard

A comprehensive analytics platform demonstrating advanced data science skills for healthcare marketing optimization. This project showcases all the key requirements for the UPMC Associate Data Scientist position.

## 🎯 Project Overview

This application demonstrates:
- **Media Mix Modeling** for marketing channel effectiveness analysis
- **Multi-Touch Attribution** for patient journey optimization
- **Predictive Forecasting** for ROI and budget optimization
- **Healthcare Analytics** with patient acquisition focus
- **SQL Database Management** with complex queries
- **Interactive Visualizations** using Plotly and Streamlit

## 🏥 Healthcare Marketing Focus

The project simulates a real healthcare marketing environment with:
- Patient acquisition campaigns across multiple specialties
- Healthcare-specific metrics (patient value, specialty performance)
- Marketing channels relevant to healthcare (Digital, Search, Social, Email, TV, Radio, Print)
- Patient journey analysis from awareness to conversion
- ROI optimization for healthcare marketing budgets

## 📊 Key Features

### 1. Media Mix Modeling
- **Econometric analysis** of marketing channel contributions
- **Adstock transformation** for carryover effects
- **Elasticity analysis** for budget optimization
- **Channel contribution** measurement and attribution

### 2. Multi-Touch Attribution
- **Shapley-based attribution** for fair credit assignment
- **Position-based weighting** (first-touch, last-touch, middle-touch)
- **Patient journey analysis** with touchpoint sequencing
- **Attribution lift analysis** comparing models

### 3. ROI Forecasting
- **Ensemble modeling** (Random Forest, Gradient Boosting, Linear Regression)
- **Budget optimization** scenarios and recommendations
- **Scenario analysis** for different market conditions
- **Predictive accuracy** with cross-validation

### 4. Interactive Dashboard
- **Executive overview** with key performance indicators
- **Channel performance** analysis and visualization
- **Patient journey** flow and pattern analysis
- **Budget allocation** optimization recommendations

### 5. Database Management
- **SQLite database** with healthcare marketing schema
- **Complex SQL queries** for data analysis
- **Data quality** metrics and validation
- **Custom query interface** for ad-hoc analysis

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd upmcads
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run main.py
   ```

4. **Access the dashboard**
   - Open your browser to `http://localhost:8501`
   - Navigate through the different analysis sections

## 📁 Project Structure

```
upmcads/
├── main.py                     # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
├── database/
│   ├── __init__.py
│   └── db_manager.py          # SQL database management
├── models/
│   ├── __init__.py
│   ├── media_mix_model.py     # Media Mix Modeling
│   ├── attribution_model.py   # Multi-Touch Attribution
│   └── forecasting_model.py   # ROI Forecasting
└── visualization/
    ├── __init__.py
    └── dashboard.py           # Interactive Dashboard
```

## 🔬 Technical Implementation

### Media Mix Modeling
- **Ridge Regression** for handling multicollinearity
- **Adstock transformation** for carryover effects
- **Feature engineering** with seasonal and lag variables
- **Cross-validation** for model performance evaluation

### Attribution Analysis
- **Logistic Regression** for conversion probability
- **Shapley values** for fair attribution
- **Position-based weighting** for journey analysis
- **Feature importance** analysis with coefficients

### Forecasting Models
- **Random Forest** for non-linear patterns
- **Gradient Boosting** for ensemble strength
- **Linear Regression** for baseline comparison
- **Ensemble averaging** for improved accuracy

### Database Design
- **Normalized schema** with foreign key relationships
- **Indexes** for query optimization
- **Data validation** and quality checks
- **Sample data generation** for realistic scenarios

## 📈 Analytics Capabilities

### Business Intelligence
- **KPI dashboards** with real-time metrics
- **Trend analysis** with time series visualization
- **Performance benchmarking** across channels
- **ROI optimization** recommendations

### Statistical Analysis
- **Hypothesis testing** for channel effectiveness
- **Confidence intervals** for predictions
- **Elasticity calculations** for budget planning
- **Correlation analysis** for variable relationships

### Predictive Analytics
- **Time series forecasting** for future performance
- **Scenario modeling** for what-if analysis
- **Budget optimization** with constraint handling
- **Risk assessment** for investment decisions

## 🎨 Visualization Features

### Interactive Charts
- **Plotly** for professional interactive visualizations
- **Real-time updates** with user selections
- **Drill-down capability** for detailed analysis
- **Export functionality** for reporting

### Dashboard Components
- **Executive summary** with key metrics
- **Channel performance** comparison
- **Patient journey** flow analysis
- **Budget allocation** optimization

## 💼 UPMC Position Alignment

### Required Skills Demonstrated

#### Advanced Analytics
- ✅ **Media Mix Modeling** - Econometric analysis of marketing effectiveness
- ✅ **Multi-Touch Attribution** - Patient journey optimization
- ✅ **Predictive Forecasting** - ROI and budget optimization

#### Technical Skills
- ✅ **Python Programming** - Object-oriented design with advanced libraries
- ✅ **SQL Database Management** - Complex queries and data architecture
- ✅ **Statistical Modeling** - Multiple regression techniques
- ✅ **Data Visualization** - Interactive dashboards and reporting

#### Healthcare Focus
- ✅ **Healthcare Analytics** - Patient acquisition and specialty analysis
- ✅ **Marketing Analytics** - Channel performance and ROI optimization
- ✅ **Business Intelligence** - Executive dashboards and KPIs

#### Problem-Solving
- ✅ **Data Architecture** - Scalable database design
- ✅ **Model Validation** - Cross-validation and performance metrics
- ✅ **Business Impact** - Actionable insights and recommendations

## 📊 Sample Insights

The application generates insights such as:
- "Search Engine marketing has 85% elasticity - highly responsive to budget changes"
- "Cardiology campaigns show 23% higher ROI than average"
- "Multi-touch attribution reveals 15% higher value for Social Media vs last-touch"
- "Optimizing budget allocation could increase ROI by 12%"

## 🔧 Customization

### Adding New Channels
1. Update `database/db_manager.py` to include new channel types
2. Modify model features in each analysis module
3. Update visualization labels and colors

### Adding New Specialties
1. Extend the specialties list in data generation
2. Update filtering options in the dashboard
3. Add specialty-specific analysis if needed

### Extending Analytics
1. Add new model classes in the `models/` directory
2. Import and integrate in `main.py`
3. Create corresponding dashboard sections

## 📞 Contact Information

**Project Author**: Portfolio submission for UPMC Associate Data Scientist role  
**Focus**: Healthcare Marketing Analytics and Data Science  
**Skills Demonstrated**: Media Mix Modeling, Attribution Analysis, Forecasting, SQL, Python, Visualization

---

*This project demonstrates production-ready code with healthcare marketing focus, showcasing all technical requirements for the UPMC Associate Data Scientist position.* 
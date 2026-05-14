# Student Placement and Salary Prediction System

## Overview

This project is a comprehensive machine learning system designed to predict student placement outcomes and expected salaries based on academic performance and co-curricular activities. The system employs advanced algorithms to address class imbalance and provides an interactive web interface for real-time predictions.

## Project Scope

The project addresses two interconnected prediction tasks in the educational domain:

1. **Placement Classification**: Predicts whether a student will be placed (hired by a company) or not placed based on academic performance and extracurricular activities.

2. **Salary Regression**: Predicts the expected salary package for students who are placed, treating salary prediction as a regression problem.

### Business Context

Campus placement is a critical milestone for engineering students. Universities and students benefit from understanding which factors contribute to successful placements and higher salaries. This predictive system enables:

- Students to identify areas for improvement
- Placement cells to provide targeted guidance
- Institutions to refine curriculum based on industry demands

## Features

- **Placement Prediction**: Binary classification using XGBoost with 94% accuracy
- **Salary Prediction**: Regression using Gradient Boosting with R² score of 0.94 and RMSE of ₹71,640
- **Interactive Web Interface**: Streamlit-based application for easy predictions
- **What-If Analysis**: Real-time scenario testing with adjustable parameters
- **Bias Mitigation**: SMOTE oversampling and autoencoder-based data augmentation
- **Comprehensive Evaluation**: Multiple ML algorithms compared and evaluated

## Dataset

The system uses two CSV datasets containing 10,000 student records each:

- **Placement_Prediction_data.csv**: Features and placement status
- **Salary_prediction_data.csv**: Features, placement status, and salary

### Features Used:
- CGPA (Cumulative Grade Point Average)
- Major Projects, Mini Projects
- Workshops/Certifications
- Technical Skills count
- Communication Skills Rating
- Internship Experience
- Hackathon Participation
- 12th and 10th Percentage
- Number of Backlogs

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Dependencies

Install the required packages using pip:

```bash
pip install streamlit scikit-learn xgboost pandas numpy pickle-mixin
```

For development and analysis:
```bash
pip install matplotlib seaborn imbalanced-learn tensorflow
```

## Usage

### Running the Web Application

1. Ensure the model files (`placement_model.pkl` and `salary_model.pkl`) are in the same directory as `streamlit_app.py`

2. Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

3. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

4. Enter student details in the form and click "Predict" to get placement and salary predictions

### What-If Analysis

The application includes interactive sliders to modify input parameters and observe how they affect predictions in real-time, helping students understand which improvements would most impact their placement prospects.

## Project Structure

```
├── ADS_Final.ipynb              # Jupyter notebook with complete analysis and model training
├── Placement_Prediction.py      # Script for training placement prediction model
├── Salary_prediction.py         # Script for training salary prediction model
├── streamlit_app.py             # Streamlit web application
├── project-guide.md             # Comprehensive technical documentation
├── placement_model.pkl          # Serialized placement prediction model
├── salary_model.pkl             # Serialized salary prediction model
├── placement_scaler.pkl         # Feature scaler for placement model
├── salary_scaler.pkl            # Feature scaler for salary model
└── README.md                    # This file
```

## Machine Learning Models

### Classification Models (Placement Prediction)

| Algorithm | Accuracy | F1-Score |
|-----------|----------|----------|
| XGBoost (Selected) | 94% | 0.94 |
| Random Forest | 95% | 0.95 |
| Gradient Boosting | 95% | 0.95 |
| Logistic Regression | 93% | 0.93 |
| SVM | 93% | 0.94 |
| Decision Tree | 93% | 0.93 |

### Regression Models (Salary Prediction)

| Algorithm | RMSE (₹) | R² Score |
|-----------|----------|----------|
| Gradient Boosting (Selected) | 71,640 | 0.94 |
| Random Forest | 74,625 | 0.94 |
| XGBoost | 78,111 | 0.93 |
| Decision Tree | 99,132 | 0.89 |
| Linear Regression | 127,266 | 0.81 |

## Technical Implementation

### Data Preprocessing
- Label encoding for categorical variables
- Feature standardization using StandardScaler
- SMOTE oversampling for class imbalance
- Autoencoder-based data augmentation for regression

### Model Training
- 80-20 train-test split
- Hyperparameter tuning
- Cross-validation for robust evaluation
- Model serialization with pickle

### Deployment
- Streamlit framework for web interface
- Real-time prediction pipeline
- Interactive visualization components

## Results and Performance

### Final Model Performance
- **Placement Prediction**: 94% accuracy with balanced precision/recall
- **Salary Prediction**: R² = 0.94, RMSE = ₹71,640

### Key Findings
1. CGPA, internship experience, and hackathon participation are top predictors
2. SMOTE effectively balanced classes without sacrificing accuracy
3. Gradient boosting methods outperformed linear models
4. Autoencoder augmentation improved regression performance

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Enhancements

- Incorporate time-series data for trend analysis
- Add company/industry-specific predictions
- Implement NLP for automatic skills extraction
- Deploy with monitoring and automated retraining
- Add user authentication and data persistence

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built as part of Applied Data Science coursework
- Utilizes scikit-learn, XGBoost, and Streamlit frameworks
- Dataset provided for educational purposes

## Contact

For questions or feedback, please open an issue in the repository.
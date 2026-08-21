# 🎮 GameSense AI

## Player Churn Prediction Using Machine Learning

GameSense AI is a modular machine learning project designed to predict whether a player is likely to churn based on their gaming behavior and player telemetry.

The project demonstrates a complete machine learning workflow, including data preprocessing, feature engineering, model training, model evaluation, prediction, automated testing, and Git-based version control.

---

## 📌 Project Overview

Modern online games generate large amounts of player behavioral data such as session activity, gameplay frequency, player lifetime, and recent engagement.

Understanding this data can help gaming companies identify players who may be at risk of leaving the game.

**GameSense AI** uses machine learning classification models to analyze player behavior and predict the probability of player churn.

### Core Question

> Can we predict whether a player is likely to churn based on their gameplay behavior?

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze player behavioral and telemetry data.
- Clean and preprocess raw gaming data.
- Remove irrelevant and constant features.
- Engineer meaningful player engagement features.
- Split data into training and testing sets.
- Train multiple machine learning classification models.
- Compare model performance.
- Select the best-performing model.
- Evaluate the final model using multiple metrics.
- Build a reusable prediction pipeline.
- Implement automated testing.
- Maintain a modular and version-controlled ML repository.

---

## 🧠 Machine Learning Problem

This project is formulated as a **binary classification problem**.

The target variable is:

`player_churn`

The target contains two classes:

| Value | Meaning |
|---|---|
| `0` / `False` | Player is not churned |
| `1` / `True` | Player is churned |

The model learns patterns from player behavior and predicts whether a player belongs to the churned class.

---

## 📊 Dataset

The project uses a player telemetry dataset containing behavioral information collected from game activity.

### Dataset Size

- **2,096 player records**
- **187 original features**

The dataset contains information related to:

- Player lifetime
- Session activity
- Stage activity
- Recent gameplay behavior
- Weekly activity
- Monthly activity
- Time-of-day gameplay patterns
- Player cohort information
- Player type
- Churn status

### Important Variables

Some important variables include:

- `player_id`
- `cohort_id`
- `cohort_day_of_week`
- `player_type`
- `player_lifetime`
- `session_count`
- `player_churn`

along with historical session, stage, and time-of-day telemetry features.

---

## 🔍 Dataset Analysis

Initial analysis of the dataset showed:

- Total records: **2,096**
- Original features: **187**
- Not churned: **1,174**
- Churned: **922**

The target distribution is reasonably balanced, allowing us to train classification models without severe class imbalance.

The dataset also contains missing values in several behavioral telemetry features.

These missing values can occur naturally when a player has no recorded activity during a particular period.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline performs several important operations.

### 1. Remove Player Identifiers

The `player_id` column is an identifier rather than a meaningful predictive feature.

Therefore:

`player_id → removed`

This prevents the model from learning meaningless patterns from unique player identifiers.

### 2. Remove Constant Features

Features containing only one unique value are removed automatically.

These features contain no useful predictive information.

### 3. Handle Missing Values

Numerical features use median imputation.

Categorical features use most-frequent-value imputation.

### 4. Encode Categorical Variables

Categorical variables are converted into numerical representations using `OneHotEncoder`.

### 5. Prevent Unknown Category Errors

The encoder uses `handle_unknown="ignore"` so the prediction pipeline can handle categories that were not present during training.

---

## 🧪 Feature Engineering

Feature engineering is used to transform raw telemetry into meaningful player engagement indicators.

### Total Session Activity

Aggregates session activity across available historical telemetry periods.

`total_session_activity`

### Total Stage Activity

Aggregates stage-level activity.

`total_stage_activity`

### Session Activity Relative to Player Lifetime

A normalized measure of activity relative to player lifetime.

`sessions_per_lifetime`

These features are designed to capture player engagement patterns rather than relying only on individual raw telemetry columns.

---

## ⚠️ Target Leakage Consideration

During dataset analysis, `player_type` was found to have a very strong relationship with the target variable `player_churn`.

The `player_type` column contains categories such as:

- `churner`
- `casual`
- `hardcore`

Because the `churner` category is directly associated with the churn outcome, using `player_type` as a model feature could introduce target leakage.

Therefore:

`player_type → excluded from model training`

This makes the prediction problem more meaningful and prevents the model from simply learning information that is too closely related to the target.

---

## 🤖 Machine Learning Models

Multiple classification algorithms are trained and compared.

### 1. Logistic Regression

Used as a baseline classification model.

Advantages:

- Simple
- Interpretable
- Fast
- Useful baseline for comparison

### 2. Random Forest

An ensemble of decision trees.

Advantages:

- Handles nonlinear relationships
- Works well with structured/tabular data
- Robust to noisy features
- Can capture complex feature interactions

### 3. Gradient Boosting

An ensemble learning technique that builds models sequentially to correct previous errors.

Advantages:

- Strong predictive performance
- Captures nonlinear relationships
- Effective for structured datasets

---

## 📈 Model Evaluation

The models are evaluated using multiple metrics.

### Accuracy

Measures the overall proportion of correct predictions.

`Accuracy = Correct Predictions / Total Predictions`

### Precision

Measures how many predicted churners were actually churners.

`Precision = TP / (TP + FP)`

### Recall

Measures how many actual churners were successfully identified.

`Recall = TP / (TP + FN)`

### F1 Score

Balances precision and recall.

`F1 = 2 × (Precision × Recall) / (Precision + Recall)`

### ROC-AUC

Measures the model's ability to distinguish between churned and non-churned players across different classification thresholds.

### Confusion Matrix

Used to visualize:

- True Positives
- True Negatives
- False Positives
- False Negatives

---

## 🏗️ Project Architecture

The GameSense AI pipeline follows this architecture:

Player Telemetry  
↓  
Raw Dataset (CSV)  
↓  
Data Preprocessing  
↓  
Feature Engineering  
↓  
Train/Test Split  
↓  
Model Training  
↓  
Logistic Regression + Random Forest + Gradient Boosting  
↓  
Model Evaluation  
↓  
Best Performing Model  
↓  
Prediction Pipeline  
↓  
Churn Probability  
↓  
Low / Medium / High Risk

---

## 📁 Project Structure

gamesense-ai/
│
├── data/
│   ├── raw/
│   │   └── player-churn.csv
│   │
│   └── processed/
│
├── models/
│   ├── model_comparison.csv
│   └── confusion_matrix.png
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py
│
├── requirements.txt
├── README.md
└── .gitignore

---

## 🔄 Machine Learning Pipeline

The complete workflow is:

1. Load Raw Dataset
2. Remove Unnecessary Features
3. Handle Missing Values
4. Encode Categorical Variables
5. Perform Feature Engineering
6. Split Data into Training and Testing Sets
7. Train Multiple Models
8. Evaluate Models
9. Select Best Model
10. Save Best Model
11. Generate Predictions

---

## 🧪 Testing

Automated tests are included using `pytest`.

The tests verify:

- Dataset can be loaded.
- Dataset is not empty.
- Target variable exists.
- Identifier columns are removed correctly.

Run the tests using:

`pytest`

---

## 🚀 How to Run the Project

### 1. Clone the Repository

`git clone https://github.com/rud11fadte/gamesense-ai.git`

Move into the project directory:

`cd gamesense-ai`

### 2. Create a Virtual Environment

`python -m venv .venv`

### 3. Activate the Virtual Environment

On Windows PowerShell:

`.venv\Scripts\Activate.ps1`

### 4. Install Dependencies

`pip install -r requirements.txt`

### 5. Train the Models

`python src/train.py`

The training pipeline will:

- Load the dataset
- Preprocess the data
- Engineer features
- Split training and testing data
- Train multiple classification models
- Compare model performance
- Select the best model
- Save model evaluation results

### 6. Evaluate the Best Model

`python src/evaluate.py`

This generates:

- Classification report
- Confusion matrix
- Evaluation visualization

### 7. Run Automated Tests

`pytest`

---

## 🎮 Real-World Use Cases

GameSense AI can potentially be used by gaming companies for several applications.

### 1. Player Retention

Identify players who are likely to stop playing.

Player Behavior  
↓  
ML Model  
↓  
High Churn Risk  
↓  
Retention Campaign

Gaming companies could then provide targeted:

- Rewards
- Missions
- Events
- Bonuses
- Personalized content

### 2. Personalized Player Experience

The model could help identify players who may need additional engagement and allow game systems to provide more personalized experiences.

### 3. Marketing Optimization

Instead of targeting every player, gaming companies could focus retention campaigns on players predicted to have a high probability of churn.

### 4. Game Design Analysis

Churn predictions can help developers investigate whether certain gameplay patterns or engagement levels are associated with players leaving the game.

### 5. Live-Service Games

The system could be incorporated into games with:

- Seasonal content
- Battle passes
- Ranked systems
- Live events
- New maps
- New characters
- Continuous content updates

---

## 💡 Future Improvements

The current project provides the core machine learning pipeline.

Future improvements could include:

- Interactive Streamlit dashboard
- SHAP-based model explainability
- Player risk visualization
- Real-time player prediction
- Player segmentation using K-Means
- Hyperparameter optimization
- XGBoost model
- MLflow experiment tracking
- Automated model retraining
- Docker deployment
- REST API for predictions
- Cloud deployment

---

## 🖥️ Planned GameSense AI Dashboard

A future version of the project can provide an interactive gaming analytics dashboard containing:

- Total players
- Churn rate
- Average sessions
- Player engagement trends
- Churn risk distribution
- Player activity
- Individual player risk analysis
- Churn probability
- Model explanations

The dashboard will provide an easy-to-understand interface for interpreting the machine learning results.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| Matplotlib | Data visualization |
| Seaborn | Data visualization |
| Joblib | Model serialization |
| Pytest | Automated testing |
| Git | Version control |
| GitHub | Repository hosting |
| VS Code | Development environment |

---

## 🌿 Git Workflow

The project follows a feature-branch development workflow.

Branches used during development include:

- `main`
- `feature/project-setup`
- `feature/data-preprocessing`
- `feature/feature-engineering`

Additional development branches can be created for future functionality:

- `feature/model-training`
- `feature/model-evaluation`
- `feature/prediction`
- `feature/testing`
- `feature/dashboard`

This workflow keeps individual features isolated and makes the development history easier to understand.

---

## 📝 Commit Convention

Meaningful commit messages are used throughout the project.

Examples:

- `chore: initialize project structure`
- `feat: add data preprocessing pipeline`
- `feat: implement feature engineering`
- `feat: add model training pipeline`
- `feat: add model evaluation`
- `test: add preprocessing tests`
- `docs: update README`

The commit history follows a conventional style to make changes clear and maintainable.

---

## 📌 Project Status

### Completed

- [x] Project structure created
- [x] Dataset added
- [x] Virtual environment configured
- [x] Git repository initialized
- [x] Feature branches created
- [x] Data preprocessing implemented
- [x] Feature engineering implemented
- [x] Model training pipeline implemented
- [x] Multiple ML models implemented
- [x] Model evaluation implemented
- [x] Prediction module implemented
- [x] Automated tests added
- [x] README documentation created
- [x] GitHub repository created

### Planned

- [ ] Interactive Streamlit dashboard
- [ ] SHAP model explainability
- [ ] Advanced visualizations
- [ ] Hyperparameter optimization
- [ ] MLflow experiment tracking
- [ ] API deployment
- [ ] Cloud deployment

---

## 📚 Academic Learning Outcomes

This project demonstrates practical understanding of:

- Supervised Machine Learning
- Binary Classification
- Data Preprocessing
- Missing Value Handling
- Feature Engineering
- Feature Selection
- Target Leakage
- Train/Test Splitting
- Model Comparison
- Model Evaluation
- Pipeline Construction
- Automated Testing
- Modular Programming
- Git and GitHub
- Feature Branch Development
- ML Project Organization

---

## 👨‍💻 Author

**Rudresh Fadate**

MSc Data Science  
Goa University
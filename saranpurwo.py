import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('train.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Check unique values in target
print("\nUnique Fertilizer Names:", df['Fertilizer Name'].nunique())
print("\nClass distribution:")
print(df['Fertilizer Name'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Preprocessing
# Convert categorical features to numerical
label_encoders = {}
categorical_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"{col} classes:", le.classes_)

# Features and target
X = df.drop(['id', 'Fertilizer Name'], axis=1)
y = df['Fertilizer Name']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Import all models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42),
    'CatBoost': cb.CatBoostClassifier(n_estimators=100, random_state=42, verbose=0),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Bagging': BaggingClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate individual models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Print classification report for top models
    if accuracy > 0.8:
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=label_encoders['Fertilizer Name'].classes_))

# Create ensemble models
print("\n" + "="*50)
print("ENSEMBLE MODELS")
print("="*50)

# 1. Voting Classifier (Hard Voting)
print("\n1. Voting Classifier (Hard Voting)...")
voting_hard = VotingClassifier(
    estimators=[
        ('rf', models['Random Forest']),
        ('xgb', models['XGBoost']),
        ('lgb', models['LightGBM']),
        ('cat', models['CatBoost'])
    ],
    voting='hard'
)
voting_hard.fit(X_train_scaled, y_train)
y_pred_voting = voting_hard.predict(X_test_scaled)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
results['Voting Classifier (Hard)'] = accuracy_voting
print(f"Voting Classifier (Hard) Accuracy: {accuracy_voting:.4f}")

# 2. Voting Classifier (Soft Voting)
print("\n2. Voting Classifier (Soft Voting)...")
voting_soft = VotingClassifier(
    estimators=[
        ('rf', models['Random Forest']),
        ('xgb', models['XGBoost']),
        ('lgb', models['LightGBM']),
        ('cat', models['CatBoost']),
        ('svm', models['Support Vector Machine'])
    ],
    voting='soft'
)
voting_soft.fit(X_train_scaled, y_train)
y_pred_soft = voting_soft.predict(X_test_scaled)
accuracy_soft = accuracy_score(y_test, y_pred_soft)
results['Voting Classifier (Soft)'] = accuracy_soft
print(f"Voting Classifier (Soft) Accuracy: {accuracy_soft:.4f}")

# 3. Stacking Classifier
print("\n3. Stacking Classifier...")
stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')),
        ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)
stacking.fit(X_train_scaled, y_train)
y_pred_stack = stacking.predict(X_test_scaled)
accuracy_stack = accuracy_score(y_test, y_pred_stack)
results['Stacking Classifier'] = accuracy_stack
print(f"Stacking Classifier Accuracy: {accuracy_stack:.4f}")

# Display all results
print("\n" + "="*50)
print("FINAL RESULTS COMPARISON")
print("="*50)

# Sort results by accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("\nModel Performance (sorted by accuracy):")
print("-" * 50)
for name, accuracy in sorted_results:
    print(f"{name:30s}: {accuracy:.4f}")

# Visualize results
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 8))
names = [name for name, _ in sorted_results]
accuracies = [acc for _, acc in sorted_results]

bars = plt.barh(names, accuracies, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Model Comparison for Fertilizer Recommendation')
plt.xlim([0, 1])
plt.grid(axis='x', alpha=0.3)

# Add accuracy values on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{acc:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()

# Feature importance from best model
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Use the best model (XGBoost or Random Forest)
best_model_name = sorted_results[0][0]
if best_model_name in ['XGBoost', 'Random Forest', 'LightGBM', 'CatBoost']:
    best_model = models[best_model_name] if best_model_name in models else stacking
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Important Features from {best_model_name}:")
        print(feature_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'].head(15), 
                feature_importance['importance'].head(15))
        plt.xlabel('Importance')
        plt.title(f'Top 15 Feature Importance - {best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

# Cross-validation for the best models
from sklearn.model_selection import cross_val_score

print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS (Top 3 Models)")
print("="*50)

top_models = sorted_results[:3]
for name, _ in top_models:
    if name in models:
        model = models[name]
    elif name == 'Stacking Classifier':
        model = stacking
    elif 'Voting' in name:
        model = voting_soft if 'Soft' in name else voting_hard
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"\n{name}:")
    print(f"  CV Scores: {cv_scores}")
    print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Create a prediction function
def recommend_fertilizer(input_features):
    """
    Function to recommend fertilizer based on input features
    input_features: dictionary with keys matching the dataset columns
    """
    # Create DataFrame from input
    input_df = pd.DataFrame([input_features])
    
    # Encode categorical variables
    for col in ['Soil Type', 'Crop Type']:
        if col in input_df.columns:
            # Handle unseen labels
            if input_df[col].iloc[0] in label_encoders[col].classes_:
                input_df[col] = label_encoders[col].transform(input_df[col])
            else:
                # Use most frequent class if unseen
                input_df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
    
    # Make sure all columns are present and in correct order
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[X.columns]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Predict using the best model
    prediction = stacking.predict(input_scaled)[0]
    prediction_proba = stacking.predict_proba(input_scaled)[0]
    
    # Get fertilizer name
    fertilizer_name = label_encoders['Fertilizer Name'].inverse_transform([prediction])[0]
    
    # Get top 3 recommendations
    top_3_idx = np.argsort(prediction_proba)[-3:][::-1]
    top_3_fertilizers = label_encoders['Fertilizer Name'].inverse_transform(top_3_idx)
    top_3_probs = prediction_proba[top_3_idx]
    
    return {
        'recommended_fertilizer': fertilizer_name,
        'confidence': max(prediction_proba),
        'top_3_recommendations': list(zip(top_3_fertilizers, top_3_probs))
    }

# Test the prediction function
print("\n" + "="*50)
print("TEST PREDICTION FUNCTION")
print("="*50)

# Example input
test_input = {
    'Temparature': 30,
    'Humidity': 65,
    'Moisture': 50,
    'Soil Type': 'Clayey',
    'Crop Type': 'Paddy',
    'Nitrogen': 25,
    'Potassium': 10,
    'Phosphorous': 20
}

recommendation = recommend_fertilizer(test_input)
print("\nExample Recommendation:")
print(f"Recommended Fertilizer: {recommendation['recommended_fertilizer']}")
print(f"Confidence: {recommendation['confidence']:.2%}")
print("\nTop 3 Recommendations:")
for fert, prob in recommendation['top_3_recommendations']:
    print(f"  {fert}: {prob:.2%}")

# Save the best model and preprocessing objects
import joblib

model_data = {
    'model': stacking,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'features': X.columns.tolist()
}

joblib.dump(model_data, 'fertilizer_recommender.pkl')
print("\nModel saved as 'fertilizer_recommender.pkl'")

# Hyperparameter tuning for best model (optional)
print("\n" + "="*50)
print("HYPERPARAMETER TUNING (Optional - XGBoost Example)")
print("="*50)

from sklearn.model_selection import GridSearchCV

# Only run if you want to optimize further
run_tuning = False  # Set to True to run tuning (takes time)

if run_tuning:
    print("Running hyperparameter tuning for XGBoost...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred_tuned = grid_search.predict(X_test_scaled)
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    print(f"Test accuracy with tuned model: {accuracy_tuned:.4f}")
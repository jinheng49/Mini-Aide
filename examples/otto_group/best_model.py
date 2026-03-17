import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import xgboost as xgb

def main():
    # Load data from './data/'
    print("Loading data...")
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')
    
    # Preprocess
    # Drop id and target columns to get features
    X = train_df.drop(['id', 'target'], axis=1).values
    y_raw = train_df['target'].values
    
    # Encode target labels (Class_1 ... Class_9 -> 0 ... 8)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Split
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=9,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred, average='macro')
    print(f"Metric: {score}")
    
    # Generate submission
    print("Generating submission...")
    test_ids = test_df['id'].values
    test_X = test_df.drop(['id'], axis=1).values
    
    test_predictions_encoded = model.predict(test_X)
    # Inverse transform to get original class names
    test_predictions = le.inverse_transform(test_predictions_encoded)
    
    submission = pd.DataFrame({
        'id': test_ids,
        'target': test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission saved!")

if __name__ == "__main__":
    main()
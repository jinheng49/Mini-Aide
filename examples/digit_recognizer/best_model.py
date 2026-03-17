import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

def main():
    # Load data from './data/'
    print("Loading data...")
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')
    
    # Preprocess
    print("Preprocessing data...")
    X = train_df.drop('label', axis=1).values / 255.0
    y = train_df['label'].values
    
    # Split
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    print("Setting up models...")
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    et_model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
    
    # Adding MLPClassifier for improved accuracy via Neural Network capabilities
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=128,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    )
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('et', et_model),
            ('mlp', mlp_model)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    # Train model
    print("Training ensemble model...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = ensemble.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Metric: {accuracy}")
    
    # Generate submission
    print("Generating submission...")
    test_predictions = ensemble.predict(test_df.values / 255.0)
    submission = pd.DataFrame({
        'ImageId': range(1, len(test_predictions) + 1),
        'Label': test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission saved!")

if __name__ == "__main__":
    main()
# model training class
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV # for model tuning
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
import os

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0

    """prepare train-test split"""
    def prepare_data(self, X, y, test_size=0.2, val_size=0.1):
        print("preparing train, validation, and test sets...")
        
        # first split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, 
            random_state=self.random_state,
            stratify=y  # 保持类别比例
        )
        
        # second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=y_train_val
        )

        print(f"train set: {X_train.shape[0]} samples")
        print(f"validation set: {X_val.shape[0]} samples")
        print(f"test set: {X_test.shape[0]} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    """initialize multiple ML models"""
    def initialize_models(self):
        self.models = {
            # traditional ML models
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),

            'Naive Bayes': MultinomialNB(),
            'SVM': LinearSVC(
                random_state=self.random_state,
                max_iter=1000
            ),

            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10
            ),

            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10
            ),

            'KNN': KNeighborsClassifier(n_neighbors=5),
            
            # advanced ensemble models
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),

            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
        
        print(f"already initialized {len(self.models)} models.")
        return self.models
    
    """train and evaluate a single model"""
    def train_single_model(self, model, name, X_train, y_train, X_val, y_val):
        print(f"\ntrain {name}...")
        
        try:
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            
            train_f1 = f1_score(y_train, y_train_pred, average='weighted')
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')
            
            # predict probabilities for AUC calculation
            if hasattr(model, 'predict_proba'):
                y_val_prob = model.predict_proba(X_val)[:, 1]   # positive class probabilities
                auc_score = roc_auc_score(y_val, y_val_prob)    # calculate AUC
            else:
                auc_score = None
            
            self.results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'auc': auc_score,
                'y_val_pred': y_val_pred,
                'y_val_true': y_val
            }
        
            print(f"  training accuracy: {train_accuracy:.4f}")
            print(f"  validation accuracy: {val_accuracy:.4f}")
            print(f"  training F1 score: {train_f1:.4f}")
            print(f"  validation F1 score: {val_f1:.4f}")
            if auc_score:
                print(f"  AUCscore: {auc_score:.4f}")
            
            # 更新最佳模型
            if val_accuracy > self.best_score:
                self.best_score = val_accuracy
                self.best_model = model
                print(f"  * the best model so far ({val_accuracy:.4f})")
            
            return model
            
        except Exception as e:
            print(f"  training failed: {e}")
            return None
    
    """train all models"""
    def train_all_models(self, X_train, y_train, X_val, y_val):
        print("\n" + "="*60)
        print("start training all models")
        print("="*60)
        
        for name, model in self.models.items():
            self.train_single_model(model, name, X_train, y_train, X_val, y_val)
        
        return self.results
    
    """cross-validation evaluation"""
    def cross_validation(self, model, X, y, cv=5):  # default 5-fold CV
        print(f"{cv}-fold cross-validation...")
        
        cv_scores = cross_val_score(
            model, X, y, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1  # use all available cores
        )
        
        print(f"cross-validation results:")
        print(f"  score of each fold: {cv_scores}")
        print(f"  average score: {cv_scores.mean():.4f}")
        print(f"  standard deviation: {cv_scores.std():.4f}")

        return cv_scores
    
    """hyperparameter tuning using GridSearchCV"""
    def hyperparameter_tuning(self, model, param_grid, X_train, y_train):
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"the best parameters: {grid_search.best_params_}")
        print(f"best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    """evaluate model on test set"""
    def evaluate_on_test(self, model, X_test, y_test):
        print("\n在测试集上评估模型...")
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"the accuracy of test set: {accuracy:.4f}")
        print(f"test set precision: {precision:.4f}")
        print(f"test set recall: {recall:.4f}")
        print(f"test set F1 score: {f1:.4f}")
        
        # confusion matrix for detailed error analysis
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nconfusion matrix:")
        print(f"           predicted")
        print(f"           0     1")
        print(f"real 0  [[{cm[0,0]:4d}  {cm[0,1]:4d}]")
        print(f"     1   [{cm[1,0]:4d}  {cm[1,1]:4d}]]")
        
        print(f"\ndetailed classification report:")
        print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
        
        if y_prob is not None:
            auc_score = roc_auc_score(y_test, y_prob)
            print(f"AUCscore: {auc_score:.4f}")
        
        test_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_true': y_test,
            'y_prob': y_prob
        }
        
        return test_results
    
    """compare all trained models"""
    def compare_models(self):
        if not self.results:
            print("no trained models to compare.")
            return
        
        print("\n" + "="*60)
        print("model comparison:")
        print("="*60)
        
        comparison_data = []
        
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Train Acc': f"{result['train_accuracy']:.4f}",
                'Val Acc': f"{result['val_accuracy']:.4f}",
                'Val F1': f"{result['val_f1']:.4f}",
                'AUC': f"{result['auc']:.4f}" if result['auc'] else 'N/A'
            })
        
        # generate comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Val Acc', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        # find and print the best model
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\nthe best model: {best_model_name}")
        print(f"accuracy test: {comparison_df.iloc[0]['Val Acc']}")
        
        return comparison_df
    
    """save trained model to disk"""
    def save_model(self, model, filename='best_model.pkl'):
        os.makedirs('models', exist_ok=True)
        filepath = os.path.join('models', filename)
        
        joblib.dump(model, filepath)
        print(f"model is saved to: {filepath}")
        
        # 同时保存模型信息
        info_file = os.path.join('models', 'model_info.txt')
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"model saving time: {pd.Timestamp.now()}\n")
            f.write(f"model type: {type(model).__name__}\n")
            if hasattr(model, 'best_params_'):
                f.write(f"best parameters: {model.best_params_}\n")
        
        return filepath
    
    """loading the best model"""
    def load_model(self, filename='best_model.pkl'):
        filepath = os.path.join('models', filename)
        
        if os.path.exists(filepath):
            model = joblib.load(filepath)
            print(f"model {filepath} loaded")
            return model
        else:
            print(f"model file does not exist: {filepath}")
            return None

# test the model trainer
if __name__ == "__main__":
    # generate testing features
    np.random.seed(42)
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 2, 100)
    
    trainer = ModelTrainer(random_state=42)
    trainer.initialize_models()
    

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        X, y, test_size=0.2, val_size=0.1
    )
    
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    comparison_df = trainer.compare_models()
    
    if trainer.best_model:
        test_results = trainer.evaluate_on_test(trainer.best_model, X_test, y_test)
    
    trainer.save_model(trainer.best_model)
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import time

class XGBoostBaseline:
    def __init__(self, params=None):
        # Default parameters from original paper or standard baseline
        self.params = params if params else {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist', # Faster on CPU/GPU
            'device': 'cuda', # Use GPU for XGBoost
            'seed': 42
        }
        self.model = xgb.XGBClassifier(**self.params)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        print("--> [INFO] Training XGBoost on GPU...")
        start_time = time.time()
        
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        end_time = time.time()
        print(f"--> [INFO] Training finished in {end_time - start_time:.2f}s")

    def evaluate(self, X_test, y_test):
        # Predict probs
        y_probs = self.model.predict_proba(X_test)[:, 1]
        y_preds = self.model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_probs)
        acc = accuracy_score(y_test, y_preds)
        f1 = f1_score(y_test, y_preds)
        
        return {
            "auc": auc,
            "accuracy": acc,
            "f1": f1,
            "y_probs": y_probs,
            "y_preds": y_preds
        }

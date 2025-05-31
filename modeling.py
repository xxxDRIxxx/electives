import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

global_model = None
X_test_global = None
y_test_global = None

def train_random_forest(df):
    global global_model, X_test_global, y_test_global

    df = df[df['RemovalType'].isin(['DIED', 'RECOVERED'])].copy()
    df['RemovalTypeEncoded'] = df['RemovalType'].map({'DIED': 1, 'RECOVERED': 0})
    df['Age'] = df['Age'].fillna(df['Age'].median())

    for col in ['Sex', 'RegionRes', 'ProvRes']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df[['Age', 'Sex', 'RegionRes', 'ProvRes']]
    y = df['RemovalTypeEncoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train_sm, y_train_sm)

    global_model = model
    X_test_global = X_test
    y_test_global = y_test

def evaluate_model():
    y_pred = global_model.predict(X_test_global)
    print(classification_report(y_test_global, y_pred, target_names=['RECOVERED', 'DIED']))

    sns.heatmap(confusion_matrix(y_test_global, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    y_prob = global_model.predict_proba(X_test_global)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_global, y_prob)
    plt.plot(fpr, tpr, label=f'AUC={auc(fpr, tpr):.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid()
    plt.show()

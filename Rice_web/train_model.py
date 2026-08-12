from django.core.management.base import BaseCommand
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class Command(BaseCommand):
    help = "Train SVM model for mobile botnet detection"

    def handle(self, *args, **options):
        DATA_PATH = "new1.csv"  # adjust path
        MODEL_OUT = "svm_MODEL.joblib"
        SCALER_OUT = "svm_scaler.joblib"

        print("Loading dataset...")
        df = pd.read_csv(DATA_PATH)
        label_col = 'Result'
        X = df.drop(columns=[label_col])
        y = df[label_col]
        X = X.fillna(0)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=1000, random_state=42)
        print("Training SVM model...")
        svm.fit(X_train_s, y_train)
        preds = svm.predict(X_test_s)

        print("\nAccuracy:", accuracy_score(y_test, preds))
        print("\nClassification Report:\n", classification_report(y_test, preds))
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("SVM Confusion Matrix")
        plt.show()

        dump(svm, MODEL_OUT)
        dump(scaler, SCALER_OUT)
        print("\nModel and scaler saved successfully!")

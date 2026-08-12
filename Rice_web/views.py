from django.contrib import messages
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from joblib import load
from PIL import Image
import cv2
import json
import numpy as np
import os
import subprocess
import sys

from .forms import NewUserForm
from .apk_utils import extract_apk_features


def index(request):
    return render(request, 'index.html')


def main(request):
    return render(request, 'main.html')


def about(request):
    return render(request, 'about.html')


def classification(request):
    if request.method == 'POST' and request.FILES.get('upload'):
        upload = request.FILES['upload']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        file_url = fss.url(file)

        img = Image.open(upload)
        img = np.array(img.convert('L'))

        grey_path = 'media/grey.jpeg'
        cv2.imwrite(grey_path, img)

        _, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
        binary_path = 'media/binary.jpeg'
        cv2.imwrite(binary_path, binary)

        return render(request, 'classification.html', {'file_url': file_url})

    return render(request, 'classification.html')


def register(request):
    if request.method == 'POST':
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Registration successful.')
            return redirect('/login1/')
        messages.error(request, 'Unsuccessful registration. Invalid information.')
    else:
        form = NewUserForm()
    return render(request=request, template_name='register.html', context={'register_form': form})


def login1(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f'You are now logged in as {username}.')
                return redirect('main')
            messages.error(request, 'Invalid username or password.')
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = AuthenticationForm()
    for name, field in form.fields.items():
        field.widget.attrs.update({'class': 'form-control', 'placeholder': name.replace('_', ' ').title()})
    return render(request=request, template_name='login.html', context={'login_form': form})


def logout_request(request):
    logout(request)
    messages.info(request, 'You have successfully logged out.')
    return redirect('login1')


_MODEL = None
_SCALER = None
_FEATURE_NAMES = None
_LABEL_MAP = None
BASE_DIR = os.path.dirname(__file__)


def _load_artifacts():
    global _MODEL, _SCALER, _FEATURE_NAMES, _LABEL_MAP

    if _MODEL is None:
        model_path = os.path.join(BASE_DIR, 'svm_MODEL.joblib')
        if not os.path.exists(model_path):
            model_path = os.path.join(BASE_DIR, 'botnet_MODEL.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model not found: {model_path}')
        _MODEL = load(model_path)

    if _SCALER is None:
        scaler_path = os.path.join(BASE_DIR, 'svm_scaler.joblib')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f'Scaler not found: {scaler_path}')
        _SCALER = load(scaler_path)

    if _FEATURE_NAMES is None:
        json_path = os.path.join(BASE_DIR, 'feature_names.json')
        joblib_path = os.path.join(BASE_DIR, 'feature_names.joblib')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                _FEATURE_NAMES = json.load(f)
        elif os.path.exists(joblib_path):
            _FEATURE_NAMES = load(joblib_path)
        else:
            raise FileNotFoundError('feature_names.json / feature_names.joblib not found')

    if _LABEL_MAP is None:
        label_path = os.path.join(BASE_DIR, 'label_mapping.json')
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                _LABEL_MAP = json.load(f)
        else:
            _LABEL_MAP = {'botnet_value': 1}

    return _MODEL, _SCALER, _FEATURE_NAMES, _LABEL_MAP


def _save_apk(uploaded_file):
    upload_dir = os.path.join('media', 'apk_uploads')
    os.makedirs(upload_dir, exist_ok=True)
    storage = FileSystemStorage(location=upload_dir, base_url='/media/apk_uploads/')
    saved_name = storage.save(uploaded_file.name, uploaded_file)
    saved_path = os.path.join(storage.location, saved_name)
    saved_url = storage.url(saved_name)
    return saved_path, saved_url


def _predict_vector(model, scaler, label_map, vector):
    X = np.array(vector, dtype=float).reshape(1, -1)
    try:
        Xs = scaler.transform(X)
    except Exception:
        Xs = X

    pred = model.predict(Xs)[0]

    score = None
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(Xs)[0]
            score = float(np.max(probs))
        except Exception:
            score = None
    elif hasattr(model, 'decision_function'):
        try:
            dec = model.decision_function(Xs)
            dec_val = float(dec[0] if hasattr(dec, '__len__') else dec)
            score = float(1.0 / (1.0 + np.exp(-abs(dec_val))))
        except Exception:
            score = None

    botnet_value = label_map.get('botnet_value', 1)
    try:
        is_botnet = int(pred) == int(botnet_value)
    except Exception:
        is_botnet = str(pred) == str(botnet_value)

    confidence = round(score * 100, 2) if score is not None else None
    if confidence is None:
        risk_level = 'Estimated'
    elif confidence >= 85:
        risk_level = 'High confidence'
    elif confidence >= 65:
        risk_level = 'Medium confidence'
    else:
        risk_level = 'Low confidence'

    return {
        'prediction_raw': pred,
        'is_botnet': is_botnet,
        'label': 'Botnet / Suspicious APK' if is_botnet else 'Normal / Benign APK',
        'subtitle': 'Potential botnet-like behavior detected' if is_botnet else 'No botnet-like signal detected',
        'color': 'danger' if is_botnet else 'success',
        'confidence': confidence,
        'risk_level': risk_level,
    }

@login_required(login_url="login1")
def detect_view(request):
    result = None
    error = None

    try:
        model, scaler, feature_names, label_map = _load_artifacts()
    except Exception as exc:
        return render(request, 'detect.html', {'error': f'Artifacts load error: {exc}'})

    if request.method == 'POST':
        uploaded = request.FILES.get('apk_file')
        if not uploaded:
            error = 'Please upload one APK file.'
        elif not uploaded.name.lower().endswith('.apk'):
            error = 'Only .apk files are allowed.'
        else:
            try:
                saved_path, saved_url = _save_apk(uploaded)
                feature_map, meta = extract_apk_features(saved_path, feature_names)
                vector = [feature_map.get(name, 0.0) for name in feature_names]
                pred_info = _predict_vector(model, scaler, label_map, vector)
                active_features = [name for name in feature_names if feature_map.get(name, 0.0) == 1.0]

                result = {
                    'file_name': uploaded.name,
                    'file_url': saved_url,
                    'package_name': meta.get('package_name', 'Unknown'),
                    'app_name': meta.get('app_name', 'Unknown'),
                    'permissions_count': meta.get('permissions_count', 0),
                    'active_features': active_features,
                    'active_count': len(active_features),
                    'feature_count': len(feature_names),
                    **pred_info,
                }
            except Exception as exc:
                error = str(exc)

    return render(request, 'detect.html', {
        'result': result,
        'error': error,
        'feature_count': len(_FEATURE_NAMES or []),
    })


# def retrain_model(request):
#     message = None
#     if request.method == 'POST':
#         script_path = os.path.join(os.path.dirname(__file__), 'train.py')
#         if not os.path.exists(script_path):
#             message = f'Script not found: {script_path}'
#         else:
#             subprocess.Popen([sys.executable, script_path])
#             message = 'Training started in background. Check your terminal output.'
#         return render(request, 'retrain.html', {'message': message})
#     return render(request, 'retrain.html', {})

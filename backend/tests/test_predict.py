import pytest
from fastapi.testclient import TestClient
from backend.main import app
import os
import shutil
 
client = TestClient(app)
 
def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get('/api/health')
        assert response.status_code == 200
        assert response.json()['status'] == 'ok'
 
def test_predict_valid_image():
    with TestClient(app) as client:
        with open('ml/data/raw/test/NORMAL/IM-0001-0001.jpeg', 'rb') as f:
            response = client.post('/api/predict', files={'file': ('test.jpg', f, 'image/jpeg')})
        assert response.status_code == 200
        data = response.json()
        assert data['prediction'] in ['NORMAL', 'PNEUMONIA']
        assert 0.0 <= data['confidence'] <= 1.0
        assert data['gradcam_image'].startswith('data:image/png;base64,')
 
def test_predict_invalid_type():
    with TestClient(app) as client:
        response = client.post('/api/predict',
            files={'file': ('test.pdf', b'fake pdf', 'application/pdf')})
        assert response.status_code == 400

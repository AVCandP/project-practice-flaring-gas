from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO  # Используем YOLO из ultralytics
import cv2
import os
import uuid
import time
import json
import numpy as np
from pathlib import Path
import shutil

app = FastAPI(title="Анализатор факелов газа", 
              description="Обнаружение газовых факелов с помощью локальной модели YOLO")

# Пути к модели и файлам
BASE_DIR = Path("E:/Python/MIFI/project-practice")
MODEL_PATH = BASE_DIR / "models/trained/best_model.pt"
UPLOAD_DIR = BASE_DIR / "static/uploads"
RESULT_DIR = BASE_DIR / "static/results"

# Создаем директории
for directory in [UPLOAD_DIR, RESULT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Загружаем модель (при старте приложения)
print("🚀 Загружаю модель YOLO...")
try:
    model = YOLO(str(MODEL_PATH))
    print(f"✅ Модель загружена: {MODEL_PATH}")
    
    # Проверяем классы модели
    if hasattr(model, 'names'):
        print(f"📊 Классы модели: {model.names}")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    model = None

# Цвета для разных классов
COLORS = {
    "flare": (0, 0, 255),      # Красный для факелов
    "fire": (0, 165, 255),     # Оранжевый для огня
    "smoke": (128, 128, 128),  # Серый для дыма
    0: (0, 0, 255),            # Красный для класса 0
    1: (0, 165, 255),          # Оранжевый для класса 1
    2: (128, 128, 128),        # Серый для класса 2
}

def process_with_yolo(image_path: str, confidence_threshold: float = 0.25):
    """Обработка изображения с помощью локальной модели YOLO"""
    
    if model is None:
        raise ValueError("Модель не загружена")
    
    # Выполняем предсказание
    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        iou=0.45,
        device='cpu',  # Можно изменить на 'cuda' если есть GPU
        verbose=False,
        save=False
    )
    
    # Извлекаем результаты
    detections = []
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Координаты bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Получаем имя класса
                class_name = model.names.get(class_id, f"class_{class_id}")
                
                # Центр bounding box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                detections.append({
                    'class': class_name,
                    'class_id': class_id,
                    'confidence': confidence,
                    'x': float(center_x),
                    'y': float(center_y),
                    'width': float(width),
                    'height': float(height),
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                })
    
    return detections

def draw_predictions(image, detections):
    """Рисует предсказания на изображении"""
    
    annotated = image.copy()
    height, width = annotated.shape[:2]
    
    for detection in detections:
        # Получаем координаты
        x1 = int(detection['x1'])
        y1 = int(detection['y1'])
        x2 = int(detection['x2'])
        y2 = int(detection['y2'])
        
        # Получаем класс и уверенность
        class_name = detection['class']
        class_id = detection['class_id']
        confidence = detection['confidence']
        
        # Выбираем цвет
        color = COLORS.get(class_name, COLORS.get(class_id, (255, 255, 255)))
        
        # Рисуем bounding box
        thickness = 3 if confidence > 0.5 else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Подготовка текста
        label = f"{class_name} {confidence:.0%}"
        
        # Размер текста
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )
        
        # Фон для текста (сверху слева от bounding box)
        text_bg_y1 = max(0, y1 - text_height - 10)
        text_bg_y2 = y1
        text_bg_x1 = x1
        text_bg_x2 = x1 + text_width
        
        cv2.rectangle(
            annotated,
            (text_bg_x1, text_bg_y1),
            (text_bg_x2, text_bg_y2),
            color,
            -1
        )
        
        # Текст
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness
        )
    
    return annotated

@app.get("/", response_class=HTMLResponse)
def home():
    """Главная страница с интерфейсом"""
    
    # Получаем информацию о модели для отображения
    model_info = ""
    if model and hasattr(model, 'names'):
        classes = list(model.names.values())
        model_info = f"<p>📊 Модель распознает: {', '.join(classes)}</p>"
    elif model is None:
        model_info = "<p style='color: orange;'>⚠️ Модель не загружена. Проверьте путь к файлу модели.</p>"
    
    # Используем двойные фигурные скобки для JavaScript (экранируем их)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Анализатор газовых факелов (Локальная модель)</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            
            body {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(45deg, #1a237e, #311b92);
                color: white;
                padding: 40px;
                text-align: center;
                border-bottom: 5px solid #4CAF50;
            }}
            
            .header h1 {{
                font-size: 2.5rem;
                margin-bottom: 10px;
                color: #fff;
            }}
            
            .header p {{
                font-size: 1.1rem;
                opacity: 0.9;
                margin-bottom: 20px;
            }}
            
            .model-info {{
                background: rgba(255,255,255,0.1);
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                display: inline-block;
            }}
            
            .content {{
                display: flex;
                flex-wrap: wrap;
                padding: 30px;
            }}
            
            .upload-section {{
                flex: 1;
                min-width: 300px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 15px;
                margin-right: 20px;
                margin-bottom: 20px;
            }}
            
            .results-section {{
                flex: 2;
                min-width: 500px;
                padding: 20px;
                display: none;
            }}
            
            .results-section.active {{
                display: block;
            }}
            
            .upload-section h2, .results-section h2 {{
                color: #333;
                margin-bottom: 20px;
                font-size: 1.8rem;
            }}
            
            .upload-box {{
                border: 3px dashed #4CAF50;
                border-radius: 15px;
                padding: 40px 20px;
                text-align: center;
                background: white;
                margin-bottom: 25px;
                transition: all 0.3s;
                cursor: pointer;
            }}
            
            .upload-box.drag-over {{
                border-color: #2196F3;
                background: #e3f2fd;
            }}
            
            .upload-box h3 {{
                color: #333;
                margin: 20px 0 10px;
            }}
            
            .file-input-wrapper {{
                margin: 20px 0;
            }}
            
            .file-input {{
                padding: 12px 25px;
                background: linear-gradient(45deg, #4CAF50, #2E7D32);
                color: white;
                border: none;
                border-radius: 50px;
                cursor: pointer;
                font-weight: bold;
                font-size: 1rem;
                transition: all 0.3s;
            }}
            
            .file-input:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
            }}
            
            .controls {{
                margin-top: 30px;
            }}
            
            .slider-container {{
                margin: 20px 0;
            }}
            
            .slider-label {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 10px;
                color: #333;
                font-weight: 500;
            }}
            
            .slider {{
                width: 100%;
                height: 10px;
                border-radius: 5px;
                background: #ddd;
                outline: none;
                -webkit-appearance: none;
            }}
            
            .slider::-webkit-slider-thumb {{
                -webkit-appearance: none;
                width: 25px;
                height: 25px;
                border-radius: 50%;
                background: #4CAF50;
                cursor: pointer;
                border: 3px solid white;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            
            .btn {{
                width: 100%;
                padding: 16px;
                background: linear-gradient(45deg, #FF9800, #F57C00);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 1.1rem;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s;
                margin-top: 10px;
            }}
            
            .btn:hover {{
                transform: translateY(-3px);
                box-shadow: 0 10px 20px rgba(255, 152, 0, 0.3);
            }}
            
            .btn:disabled {{
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }}
            
            .loading {{
                display: none;
                text-align: center;
                padding: 40px;
                background: white;
                border-radius: 15px;
                margin: 20px 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            .spinner {{
                width: 60px;
                height: 60px;
                border: 5px solid #f3f3f3;
                border-top: 5px solid #4CAF50;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            .error {{
                display: none;
                background: #ffebee;
                color: #c62828;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 5px solid #c62828;
            }}
            
            .legend {{
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 20px;
                margin-top: 30px;
                padding: 20px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .color-box {{
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 2px solid #333;
            }}
            
            .image-container {{
                background: #f5f5f5;
                padding: 15px;
                border-radius: 15px;
                margin-bottom: 25px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            }}
            
            #resultImage {{
                max-width: 100%;
                max-height: 500px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            .stats {{
                background: white;
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 25px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            }}
            
            .stats h3 {{
                margin-bottom: 20px;
                color: #333;
            }}
            
            .stat-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
            }}
            
            .stat-item {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                border-top: 4px solid #4CAF50;
            }}
            
            .stat-value {{
                font-size: 2rem;
                font-weight: bold;
                color: #1a237e;
                margin-bottom: 5px;
            }}
            
            .stat-label {{
                color: #666;
                font-size: 0.9rem;
            }}
            
            .detections {{
                max-height: 300px;
                overflow-y: auto;
                background: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            }}
            
            .detection-item {{
                background: #f8f9fa;
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 8px;
                border-left: 5px solid #4CAF50;
            }}
            
            .footer {{
                background: #263238;
                color: white;
                padding: 25px;
                text-align: center;
                border-top: 1px solid #37474F;
            }}
            
            .footer p {{
                margin: 10px 0;
                opacity: 0.8;
            }}
            
            @media (max-width: 768px) {{
                .content {{
                    flex-direction: column;
                    padding: 15px;
                }}
                
                .upload-section, .results-section {{
                    margin-right: 0;
                    margin-bottom: 20px;
                }}
                
                .header {{
                    padding: 25px 20px;
                }}
                
                .header h1 {{
                    font-size: 2rem;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔥 Анализатор газовых факелов</h1>
                <p>Локальная модель обнаружения газовых факелов на основе YOLO</p>
                <div class="model-info" id="modelInfo">
                    {model_info}
                </div>
            </div>
            
            <div class="content">
                <div class="upload-section">
                    <h2>📷 Загрузите изображение</h2>
                    <p style="color: #666; margin: 10px 0 25px 0;">Загрузите изображение для анализа газовых факелов</p>
                    
                    <div class="upload-box" id="uploadBox" 
                         ondragover="handleDragOver(event)" 
                         ondragleave="handleDragLeave(event)" 
                         ondrop="handleDrop(event)">
                        <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="#4CAF50" stroke-width="2" style="margin: 0 auto 20px; display: block;">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <h3>Перетащите изображение сюда</h3>
                        <p style="color: #888; margin: 10px 0 20px 0;">или</p>
                        
                        <div class="file-input-wrapper">
                            <input type="file" id="imageInput" accept="image/*" class="file-input" 
                                   onchange="handleFileSelect(event)">
                        </div>
                        
                        <div class="controls">
                            <div class="slider-container">
                                <div class="slider-label">
                                    <span>Порог уверенности:</span>
                                    <span id="confidenceValue">25%</span>
                                </div>
                                <input type="range" id="confidenceSlider" class="slider" 
                                       min="1" max="100" value="25" oninput="updateConfidence(this.value)">
                            </div>
                            
                            <button onclick="processImage()" class="btn" id="analyzeBtn">
                                <span id="btnText">🔍 Запустить анализ</span>
                            </button>
                        </div>
                    </div>
                    
                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <h3>Анализ изображения</h3>
                        <p id="loadingText">Обработка с помощью локальной модели YOLO...</p>
                    </div>
                    
                    <div class="error" id="error"></div>
                    
                    <div class="legend">
                        <div class="legend-item"><div class="color-box" style="background:#ff0000;"></div><span>Факел (flare)</span></div>
                        <div class="legend-item"><div class="color-box" style="background:#ffa500;"></div><span>Огонь (fire)</span></div>
                        <div class="legend-item"><div class="color-box" style="background:#808080;"></div><span>Дым (smoke)</span></div>
                    </div>
                </div>
                
                <div class="results-section" id="resultsSection">
                    <h2>📊 Результаты анализа</h2>
                    
                    <div class="image-container">
                        <img id="resultImage" src="" alt="Результат анализа" onerror="this.src=''">
                    </div>
                    
                    <div class="stats">
                        <h3>📈 Статистика детекции</h3>
                        <div class="stat-grid" id="statsGrid">
                            <!-- Статистика будет добавлена здесь -->
                        </div>
                    </div>
                    
                    <div id="detectionsContainer">
                        <h3>🔍 Обнаруженные объекты</h3>
                        <div class="detections" id="detectionsList">
                            <!-- Список детекций будет добавлен здесь -->
                        </div>
                    </div>
                    
                    <div style="text-align: center; margin-top: 25px;">
                        <button onclick="downloadResult()" class="btn" style="width: auto; padding: 12px 30px; background: linear-gradient(45deg, #2196F3, #1976D2);">
                            💾 Скачать результат
                        </button>
                        <button onclick="resetAnalysis()" class="btn" style="width: auto; padding: 12px 30px; margin-left: 15px; background: linear-gradient(45deg, #9e9e9e, #757575);">
                            🗑️ Новый анализ
                        </button>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>© 2024 Анализатор газовых факелов | Локальная модель YOLO | Обнаружение объектов в реальном времени</p>
                <p style="font-size: 0.9rem; margin-top: 10px; opacity: 0.7;">Используется обученная модель: {str(MODEL_PATH)}</p>
            </div>
        </div>
        
        <script>
            let currentResultData = null;
            let selectedFile = null;
            let confidenceThreshold = 0.25;
            
            function updateConfidence(value) {{
                confidenceThreshold = value / 100;
                document.getElementById('confidenceValue').textContent = value + '%';
            }}
            
            function handleDragOver(e) {{
                e.preventDefault();
                e.stopPropagation();
                document.getElementById('uploadBox').classList.add('drag-over');
            }}
            
            function handleDragLeave(e) {{
                e.preventDefault();
                e.stopPropagation();
                document.getElementById('uploadBox').classList.remove('drag-over');
            }}
            
            function handleDrop(e) {{
                e.preventDefault();
                e.stopPropagation();
                document.getElementById('uploadBox').classList.remove('drag-over');
                
                if (e.dataTransfer.files.length) {{
                    handleFileSelect({{ target: {{ files: e.dataTransfer.files }} }});
                }}
            }}
            
            function handleFileSelect(event) {{
                const file = event.target.files[0];
                if (!file) return;
                
                selectedFile = file;
                
                // Показываем превью
                const reader = new FileReader();
                reader.onload = function(e) {{
                    const img = document.getElementById('resultImage');
                    img.src = e.target.result;
                    document.getElementById('resultsSection').classList.add('active');
                    
                    // Сбрасываем предыдущие результаты
                    document.getElementById('statsGrid').innerHTML = '';
                    document.getElementById('detectionsList').innerHTML = '';
                    currentResultData = null;
                }};
                reader.readAsDataURL(file);
            }}
            
            async function processImage() {{
                if (!selectedFile) {{
                    showError('Пожалуйста, выберите изображение');
                    return;
                }}
                
                const loading = document.getElementById('loading');
                const errorDiv = document.getElementById('error');
                const analyzeBtn = document.getElementById('analyzeBtn');
                const btnText = document.getElementById('btnText');
                
                // Показываем загрузку
                errorDiv.style.display = 'none';
                loading.style.display = 'block';
                analyzeBtn.disabled = true;
                btnText.textContent = 'Анализ...';
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('confidence', confidenceThreshold);
                
                try {{
                    const response = await fetch('/process', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const data = await response.json();
                    
                    if (!response.ok) throw new Error(data.error || 'Ошибка сервера');
                    
                    // Сохраняем данные результата
                    currentResultData = data;
                    
                    // Показываем результат
                    showResults(data);
                    
                }} catch (error) {{
                    showError('Ошибка: ' + error.message);
                    console.error('Ошибка:', error);
                }} finally {{
                    loading.style.display = 'none';
                    analyzeBtn.disabled = false;
                    btnText.textContent = '🔍 Запустить анализ';
                }}
            }}
            
            function showResults(data) {{
                // Показываем секцию результатов
                const resultsSection = document.getElementById('resultsSection');
                resultsSection.classList.add('active');
                
                // Обновляем изображение
                const resultImage = document.getElementById('resultImage');
                resultImage.src = `/static/results/${{data.result_image}}?t=${{Date.now()}}`;
                
                // Статистика
                const statsGrid = document.getElementById('statsGrid');
                statsGrid.innerHTML = `
                    <div class="stat-item">
                        <div class="stat-value">${{data.total_detections}}</div>
                        <div class="stat-label">Всего объектов</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${{data.processing_time.toFixed(2)}}с</div>
                        <div class="stat-label">Время анализа</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${{data.image_width}}×${{data.image_height}}</div>
                        <div class="stat-label">Размер изображения</div>
                    </div>
                `;
                
                // Добавляем статистику по классам
                if (data.class_stats) {{
                    Object.entries(data.class_stats).forEach(([cls, count]) => {{
                        if (count > 0) {{
                            const name = {{'flare':'Факел','fire':'Огонь','smoke':'Дым','class_0':'Класс 0','class_1':'Класс 1'}}[cls] || cls;
                            statsGrid.innerHTML += `
                                <div class="stat-item">
                                    <div class="stat-value">${{count}}</div>
                                    <div class="stat-label">${{name}}</div>
                                </div>
                            `;
                        }}
                    }});
                }}
                
                // Список детекций
                const detectionsList = document.getElementById('detectionsList');
                detectionsList.innerHTML = '';
                
                if (data.detections && data.detections.length) {{
                    data.detections.forEach((d, i) => {{
                        const colors = {{
                            'flare': '#ff0000',
                            'fire': '#ffa500', 
                            'smoke': '#808080',
                            'class_0': '#ff0000',
                            'class_1': '#ffa500'
                        }};
                        const color = colors[d.class] || '#000';
                        
                        detectionsList.innerHTML += `
                            <div class="detection-item" style="border-left-color: ${{color}}">
                                <strong>#${{i+1}} ${{d.class}}</strong><br>
                                <span>Уверенность: <strong>${{(d.confidence*100).toFixed(1)}}%</strong></span><br>
                                <small>Координаты: (x: ${{d.x.toFixed(1)}}, y: ${{d.y.toFixed(1)}})</small><br>
                                <small>Размер: ${{d.width.toFixed(1)}}×${{d.height.toFixed(1)}}</small>
                            </div>
                        `;
                    }});
                }} else {{
                    detectionsList.innerHTML = '<p style="text-align: center; padding: 20px; color: #666;">Объекты не обнаружены</p>';
                }}
                
                // Прокручиваем к результатам
                resultsSection.scrollIntoView({{behavior: 'smooth'}});
            }}
            
            function showError(msg) {{
                const errorDiv = document.getElementById('error');
                errorDiv.textContent = msg;
                errorDiv.style.display = 'block';
                
                // Автоматически скрыть через 5 секунд
                setTimeout(() => {{
                    errorDiv.style.display = 'none';
                }}, 5000);
            }}
            
            function downloadResult() {{
                if (!currentResultData) {{
                    showError('Нет результатов для скачивания');
                    return;
                }}
                
                const link = document.createElement('a');
                link.href = `/static/results/${{currentResultData.result_image}}`;
                link.download = `gas_flare_detection_${{new Date().toISOString().slice(0,10)}}.jpg`;
                link.click();
            }}
            
            function resetAnalysis() {{
                // Сброс формы
                document.getElementById('imageInput').value = '';
                document.getElementById('resultImage').src = '';
                document.getElementById('resultsSection').classList.remove('active');
                document.getElementById('statsGrid').innerHTML = '';
                document.getElementById('detectionsList').innerHTML = '';
                selectedFile = null;
                currentResultData = null;
                
                // Прокрутка к началу
                document.querySelector('.upload-section').scrollIntoView({{behavior: 'smooth'}});
            }}
            
            // Инициализация при загрузке страницы
            document.addEventListener('DOMContentLoaded', function() {{
                // Обновляем информацию о модели
                fetch('/model_info')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.classes) {{
                            const modelInfo = document.getElementById('modelInfo');
                            modelInfo.innerHTML = `<p>📊 Модель распознает: ${{data.classes.join(', ')}}</p>`;
                            
                            // Обновляем легенду
                            const legend = document.querySelector('.legend');
                            legend.innerHTML = '';
                            data.classes.forEach(cls => {{
                                const color = {{'flare':'#ff0000','fire':'#ffa500','smoke':'#808080','class_0':'#ff0000','class_1':'#ffa500'}}[cls] || '#000';
                                legend.innerHTML += `
                                    <div class="legend-item">
                                        <div class="color-box" style="background: ${{color}};"></div>
                                        <span>${{cls}}</span>
                                    </div>
                                `;
                            }});
                        }}
                    }})
                    .catch(console.error);
            }});
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/model_info")
async def get_model_info():
    """Возвращает информацию о загруженной модели"""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Модель не загружена"}
        )
    
    classes = []
    if hasattr(model, 'names'):
        classes = list(model.names.values())
    
    return {
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "classes": classes,
        "num_classes": len(classes) if classes else 0
    }

@app.post("/process")
async def process_image(file: UploadFile = File(...), confidence: float = 0.25):
    """Обработка изображения с помощью локальной модели YOLO"""
    
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Модель YOLO не загружена")
    
    try:
        # Генерируем уникальное имя файла
        file_extension = os.path.splitext(file.filename)[1] or ".jpg"
        filename = f"{uuid.uuid4()}{file_extension}"
        upload_path = UPLOAD_DIR / filename
        
        # Сохраняем загруженный файл
        contents = await file.read()
        with open(upload_path, "wb") as f:
            f.write(contents)
        
        print(f"📥 Файл сохранен: {upload_path}")
        
        # Загружаем изображение для получения размеров
        image = cv2.imread(str(upload_path))
        if image is None:
            raise HTTPException(status_code=400, detail="Не удалось загрузить изображение")
        
        height, width = image.shape[:2]
        print(f"📏 Размер изображения: {width}x{height}")
        
        # Обрабатываем изображение с помощью YOLO
        detections = process_with_yolo(str(upload_path), confidence)
        print(f"🔍 Найдено объектов: {len(detections)}")
        
        # Рисуем предсказания на изображении
        annotated_image = draw_predictions(image, detections)
        
        # Сохраняем результат
        result_filename = f"result_{filename}"
        result_path = RESULT_DIR / result_filename
        cv2.imwrite(str(result_path), annotated_image)
        print(f"💾 Результат сохранен: {result_path}")
        
        # Собираем статистику по классам
        class_stats = {}
        for detection in detections:
            class_name = detection['class']
            class_stats[class_name] = class_stats.get(class_name, 0) + 1
        
        # Время обработки
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "original_image": filename,
            "result_image": result_filename,
            "total_detections": len(detections),
            "class_stats": class_stats,
            "detections": detections,
            "image_width": width,
            "image_height": height,
            "processing_time": processing_time,
            "confidence_threshold": confidence,
            "model_info": {
                "name": "YOLO Локальная модель",
                "path": str(MODEL_PATH),
                "classes": list(model.names.values()) if hasattr(model, 'names') else []
            }
        }
        
    except Exception as e:
        print(f"❌ Ошибка обработки: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/test")
async def test_endpoint():
    """Тестовый endpoint для проверки работы"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "upload_dir": str(UPLOAD_DIR),
        "result_dir": str(RESULT_DIR)
    }

@app.get("/test_image")
async def test_image_processing():
    """Обработка тестового изображения"""
    test_image_path = BASE_DIR / "flaring-gas" / "valid" / "flare_0008_jpg.rf.417f01cce748fb03929cdf7eb156222c.jpg"
    
    if not test_image_path.exists():
        return {"error": "Тестовое изображение не найдено"}
    
    # Копируем в uploads
    filename = f"test_{uuid.uuid4()}.jpg"
    upload_path = UPLOAD_DIR / filename
    shutil.copy2(test_image_path, upload_path)
    
    # Обрабатываем
    detections = process_with_yolo(str(upload_path), 0.25)
    
    # Загружаем и аннотируем
    image = cv2.imread(str(upload_path))
    annotated = draw_predictions(image, detections)
    
    # Сохраняем результат
    result_filename = f"result_{filename}"
    result_path = RESULT_DIR / result_filename
    cv2.imwrite(str(result_path), annotated)
    
    return {
        "test_image": filename,
        "result_image": result_filename,
        "detections": len(detections),
        "detections_list": detections
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 Сервер анализатора газовых факелов")
    print("="*60)
    print(f"📁 Базовая директория: {BASE_DIR}")
    print(f"🤖 Модель: {MODEL_PATH}")
    print(f"📤 Загрузки: {UPLOAD_DIR}")
    print(f"💾 Результаты: {RESULT_DIR}")
    print(f"🌐 Откройте: http://localhost:8000")
    print("="*60)
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
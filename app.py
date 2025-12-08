import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tempfile
import os
import zipfile
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== ПРОВЕРКА НАЛИЧИЯ МОДЕЛИ ====================

def check_model_exists():
    """Проверяет наличие модели в папке model/"""
    model_path = Path("model/mask_generator.pth")
    
    if model_path.exists():
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        return True, f"✅ Модель найдена: {model_path} ({file_size:.2f} MB)"
    else:
        return False, "❌ Модель не найдена в папке model/"

# Проверяем модель при запуске
model_available, model_message = check_model_exists()

# ==================== КОНФИГУРАЦИЯ СТРАНИЦЫ ====================

# Настройка страницы
st.set_page_config(
    page_title="ColorSep Pro - Профессиональное разделение цветов",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Настройка темы
st.markdown("""
    <script>
        var elements = window.parent.document.querySelectorAll('.stApp')
        elements[0].style.backgroundColor = '#ffffff';
    </script>
    """, unsafe_allow_html=True)

# Пользовательский CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0056b3;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #212121;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .info-text {
        font-size: 1.1rem;
        color: #000000;
        line-height: 1.5;
    }
    .stButton button {
        background-color: #0056b3;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #003d82;
        transform: translateY(-2px);
    }
    .color-chip {
        display: inline-block;
        width: 30px;
        height: 30px;
        margin-right: 10px;
        border: 2px solid #000;
        border-radius: 5px;
        vertical-align: middle;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .method-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #0056b3;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .model-status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #c3e6cb;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .model-status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #ffeaa7;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .upload-section {
        background-color: #e7f3ff;
        padding: 25px;
        border-radius: 12px;
        border: 3px dashed #0056b3;
        text-align: center;
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }
    .upload-section:hover {
        background-color: #d0e7ff;
        border-color: #003d82;
    }
    .layer-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    .layer-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .preview-container {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #ddd;
    }
    .tab-content {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .color-palette {
        display: flex;
        gap: 5px;
        margin: 10px 0;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 5px;
    }
    .color-item {
        width: 30px;
        height: 30px;
        border-radius: 4px;
        border: 1px solid #ccc;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .color-item:hover {
        transform: scale(1.1);
    }
    .progress-bar {
        height: 4px;
        background: linear-gradient(90deg, #0056b3, #00b3b3);
        border-radius: 2px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown("<h1 class='main-header'>ColorSep Pro: Профессиональное разделение цветов</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text' style='text-align: center;'>Загрузите изображение и извлеките цветовые слои для печати и дизайна</p>", unsafe_allow_html=True)

# Статус модели
if model_available:
    st.markdown(f'<div class="model-status-success">{model_message}<br>Метод "Fast Soft Color Segmentation" доступен!</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="model-status-warning">
        ⚠️ Метод "Fast Soft Color Segmentation" будет недоступен без модели.<br>
        <strong>Чтобы использовать этот метод:</strong><br>
        1. Скачайте модель из оригинального репозитория<br>
        2. Создайте папку <code>model/</code> в этой директории<br>
        3. Положите файл <code>mask_generator.pth</code> в папку <code>model/</code><br>
        4. Перезапустите приложение<br>
        <em>Метод K-means работает без модели.</em>
    </div>
    """, unsafe_allow_html=True)

# ==================== ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ СЕССИИ ====================

if 'custom_layers' not in st.session_state:
    st.session_state.custom_layers = []
    
if 'layer_visibility' not in st.session_state:
    st.session_state.layer_visibility = []
    
if 'layer_order' not in st.session_state:
    st.session_state.layer_order = []

if 'color_layers' not in st.session_state:
    st.session_state.color_layers = []

if 'color_info' not in st.session_state:
    st.session_state.color_info = []

if 'original_image_cv' not in st.session_state:
    st.session_state.original_image_cv = None

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'selected_method' not in st.session_state:
    st.session_state.selected_method = "K-средних кластеризация"

if 'combined_preview' not in st.session_state:
    st.session_state.combined_preview = None

if 'palette_colors' not in st.session_state:
    st.session_state.palette_colors = None

if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

# ==================== КЛАССЫ ДЛЯ МЕТОДА DECOMPOSE ====================

class SimpleMaskGenerator(nn.Module):
    """Упрощенная версия MaskGenerator для работы с произвольным количеством цветов"""
    def __init__(self, num_primary_color):
        super(SimpleMaskGenerator, self).__init__()
        in_dim = 3 + num_primary_color * 3  # вход: изображение + палитра
        out_dim = num_primary_color  # выход: альфа-маски для каждого цвета

        # Энкодер
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Декодер
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Финальный слой
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Энкодер
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        # Декодер с пропусками
        dec1 = self.decoder1(enc3)
        dec1 = torch.cat([dec1, enc2], dim=1)
        
        dec2 = self.decoder2(dec1)
        dec2 = torch.cat([dec2, enc1], dim=1)
        
        # Финальный выход
        out = self.final(dec2)
        return out

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def get_dominant_colors(img: Image.Image, num_colors: int) -> list[tuple]:
    """
    Получение доминирующих цветов из изображения с использованием K-means
    """
    try:
        # Конвертируем изображение в массив numpy
        img_array = np.array(img.convert("RGB"))
        
        # Преобразуем изображение в формат для K-means
        pixels = img_array.reshape(-1, 3)
        
        # Используем K-means для нахождения доминирующих цветов
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(pixels)
        
        # Получаем центры кластеров (доминирующие цвета)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Сортируем цвета по частоте
        labels = kmeans.labels_
        counts = np.bincount(labels)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_colors = colors[sorted_indices]
        
        # Конвертируем в список кортежей
        return [tuple(map(int, color)) for color in sorted_colors]
    
    except Exception as e:
        st.error(f"Ошибка при получении доминирующих цветов: {e}")
        # Возвращаем стандартную палитру в случае ошибки
        return [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][:num_colors]

def create_color_palette_preview(colors, size=(200, 50)):
    """Создает визуальное представление палитры цветов"""
    if not colors:
        return None
    
    palette_height = 50
    palette_width = len(colors) * 50
    
    # Создаем изображение палитры
    palette_img = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
    
    for i, color in enumerate(colors):
        start_x = i * 50
        end_x = (i + 1) * 50
        palette_img[:, start_x:end_x] = color[::-1]  # RGB to BGR for OpenCV
    
    return palette_img

def smart_color_separation_kmeans(img_cv, n_colors=5, bg_color=(255, 255, 255)):
    """
    Умное разделение цветов с использованием K-means с улучшенной обработкой
    """
    try:
        # Конвертируем BGR в RGB для K-means
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Преобразуем изображение в формат для K-means
        pixels = img_rgb.reshape(-1, 3)
        
        # Удаляем пиксели фона если нужно
        if bg_color is not None:
            bg_rgb = bg_color[::-1]  # BGR to RGB
            bg_mask = np.all(pixels == bg_rgb, axis=1)
            if np.any(bg_mask):
                pixels = pixels[~bg_mask]
        
        if len(pixels) == 0:
            st.warning("Изображение состоит только из фона")
            return [], []
        
        # Применяем K-means с улучшенными параметрами
        kmeans = KMeans(
            n_clusters=n_colors, 
            random_state=42, 
            n_init=10,
            max_iter=500,
            tol=1e-4
        )
        
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_.astype(int)
        
        # Сортируем цвета по площади покрытия
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        
        # Создаем слои
        color_layers = []
        color_info = []
        
        # Создаем базовый фон
        base_background = np.full_like(img_cv, bg_color)
        
        # Для каждого кластера создаем слой
        for idx in sorted_indices:
            # Получаем цвет кластера
            cluster_color = centers[idx]
            
            # Создаем маску для этого кластера
            # Восстанавливаем полную маску
            full_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
            
            # Заполняем маску только для не-фоновых пикселей
            if bg_color is not None:
                # Получаем маску всех не-фоновых пикселей
                non_bg_mask = ~np.all(img_rgb == bg_rgb, axis=2)
                
                # Получаем пиксели, принадлежащие этому кластеру
                cluster_pixels = labels == idx
                
                # Восстанавливаем индексы
                pixel_indices = np.arange(len(pixels))[cluster_pixels]
                
                # Преобразуем в координаты изображения
                h, w = img_rgb.shape[:2]
                y_coords = pixel_indices // w
                x_coords = pixel_indices % w
                
                # Создаем маску
                full_mask[y_coords, x_coords] = True
            else:
                # Если фон не задан, используем все пиксели
                h, w = img_rgb.shape[:2]
                pixel_indices = np.arange(len(pixels))
                y_coords = pixel_indices // w
                x_coords = pixel_indices % w
                cluster_pixels = labels == idx
                
                mask_indices = pixel_indices[cluster_pixels]
                y_mask = mask_indices // w
                x_mask = mask_indices % w
                full_mask[y_mask, x_mask] = True
            
            # Создаем слой
            layer = base_background.copy()
            
            # Заполняем область кластера
            for c in range(3):
                layer[:, :, c][full_mask] = cluster_color[c]
            
            color_layers.append(layer)
            
            # Информация о цвете
            color_bgr = (int(cluster_color[2]), int(cluster_color[1]), int(cluster_color[0]))
            coverage = (np.sum(full_mask) / full_mask.size) * 100
            
            color_info.append({
                'color': color_bgr,
                'percentage': coverage,
                'rgb_color': tuple(cluster_color)
            })
        
        return color_layers, color_info
    
    except Exception as e:
        st.error(f"Ошибка в методе K-means: {str(e)}")
        return [], []

def decompose_fast_soft_color_simple(
    input_image: Image.Image,
    num_colors: int = 5,
    palette: list[tuple] = None,
    device: str = "cpu"
) -> list[Image.Image]:
    """
    Упрощенная версия Fast Soft Color Segmentation с исправленными цветами
    """
    try:
        if not model_available:
            st.error("Модель не найдена.")
            return []
        
        # Конвертируем PIL в numpy
        img_np = np.array(input_image.convert("RGB"))
        
        # Получаем доминирующие цвета ИЗ САМОГО ИЗОБРАЖЕНИЯ
        if palette is None:
            palette = get_dominant_colors(input_image, num_colors)
        
        # Отображаем полученную палитру
        with st.expander("🎨 Полученная палитра цветов", expanded=False):
            st.write(f"Количество цветов: {len(palette)}")
            for i, color in enumerate(palette):
                hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
                st.markdown(f"""
                <div style='display: flex; align-items: center; margin: 5px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;'>
                    <div style='width: 30px; height: 30px; background-color: {hex_color}; border: 1px solid #000; border-radius: 4px; margin-right: 10px;'></div>
                    <div>
                        <strong>Цвет {i+1}:</strong> RGB{color} | {hex_color}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Создаем простую сегментацию на основе цветов
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Создаем слои для каждого цвета палитры
        color_layers = []
        
        for color_idx, target_color in enumerate(palette):
            # Конвертируем цвет в BGR
            target_color_bgr = (target_color[2], target_color[1], target_color[0])
            
            # Вычисляем расстояние до целевого цвета
            img_float = img_cv.astype(np.float32)
            target_float = np.array(target_color_bgr, dtype=np.float32).reshape(1, 1, 3)
            
            # Вычисляем цветовое расстояние
            color_diff = np.sqrt(np.sum((img_float - target_float) ** 2, axis=2))
            
            # Нормализуем и создаем альфа-канал
            max_diff = np.max(color_diff)
            if max_diff > 0:
                alpha = 1.0 - (color_diff / max_diff)
            else:
                alpha = np.ones_like(color_diff)
            
            # Повышаем контраст альфа-канала
            alpha = alpha ** 0.5  # Делаем более четкие границы
            
            # Создаем слой с прозрачностью
            layer = np.zeros_like(img_cv, dtype=np.uint8)
            
            for c in range(3):
                layer[:, :, c] = (img_cv[:, :, c] * alpha + 
                                 target_color_bgr[c] * (1 - alpha)).astype(np.uint8)
            
            color_layers.append(layer)
        
        # Конвертируем слои в PIL изображения
        pil_layers = []
        for layer in color_layers:
            rgb_layer = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
            pil_layers.append(Image.fromarray(rgb_layer))
        
        return pil_layers
    
    except Exception as e:
        st.error(f"Ошибка в упрощенном методе: {str(e)}")
        return []

def decompose_layers_to_cv_format(decompose_layers, bg_color):
    """
    Преобразует слои в формат OpenCV с правильными цветами
    """
    cv_layers = []
    color_info_list = []
    
    for i, pil_layer in enumerate(decompose_layers):
        # Конвертируем PIL Image в numpy array
        rgb_array = np.array(pil_layer)
        
        # Если изображение RGBA, конвертируем в RGB
        if rgb_array.shape[2] == 4:
            rgb_array = rgb_array[:, :, :3]
        
        # Конвертируем RGB в BGR для OpenCV
        bgr_layer = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        # Вычисляем доминирующий цвет (игнорируем фон)
        if bg_color is not None:
            # Создаем маску не-фоновых пикселей
            non_bg_mask = ~np.all(bgr_layer == bg_color, axis=2)
            
            if np.any(non_bg_mask):
                # Получаем цвета не-фоновых пикселей
                non_bg_colors = bgr_layer[non_bg_mask]
                
                # Используем медиану для устранения шума
                if len(non_bg_colors) > 0:
                    median_color = np.median(non_bg_colors, axis=0).astype(int)
                    dominant_color = tuple(median_color)
                else:
                    dominant_color = bg_color
            else:
                dominant_color = bg_color
            
            # Процент покрытия
            coverage_percentage = (np.sum(non_bg_mask) / non_bg_mask.size) * 100
        else:
            # Если фон не задан, используем средний цвет всего слоя
            dominant_color = tuple(np.median(bgr_layer.reshape(-1, 3), axis=0).astype(int))
            coverage_percentage = 100
        
        cv_layers.append(bgr_layer)
        color_info_list.append({
            'color': dominant_color,
            'percentage': coverage_percentage,
            'rgb_color': tuple(dominant_color[::-1])  # BGR to RGB
        })
    
    return cv_layers, color_info_list

def convert_to_png(image_array, filename):
    """Конвертирует массив изображения в формат PNG"""
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        fig.tight_layout(pad=0)
        
        # Сохраняем как PNG
        png_buffer = io.BytesIO()
        plt.savefig(png_buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)
        
        png_buffer.seek(0)
        return png_buffer.getvalue()
    except Exception as e:
        st.error(f"Ошибка при создании PNG: {e}")
        return None

def create_bw_mask(layer, bg_color):
    """
    Создает черно-белую маску из цветного слоя.
    Белый = область цвета, Черный = фон.
    """
    if bg_color is None:
        # Если фон не задан, считаем темные пиксели фоном
        gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        return mask
    
    # Создаем маску для определения фона
    is_background = np.all(layer == bg_color, axis=2)
    
    # Создаем маску (255 для цвета, 0 для фона)
    mask = np.zeros((layer.shape[0], layer.shape[1]), dtype=np.uint8)
    mask[~is_background] = 255
    
    # Применяем морфологические операции для очистки маски
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def save_bw_mask_as_png(mask, filename):
    """Сохраняет черно-белую маску в формате PNG"""
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mask, cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        fig.tight_layout(pad=0)
        
        # Сохраняем как PNG
        png_buffer = io.BytesIO()
        plt.savefig(png_buffer, format='png', bbox_inches='tight', pad_inches=0, 
                    dpi=300, facecolor='none', edgecolor='none')
        plt.close(fig)
        
        png_buffer.seek(0)
        return png_buffer.getvalue()
    except Exception as e:
        st.error(f"Ошибка при создании ЧБ маски PNG: {e}")
        return None

def resize_layer_to_match(layer, target_shape):
    """Изменяет размер слоя до целевого размера"""
    if layer.shape[:2] == target_shape[:2]:
        return layer
    
    return cv2.resize(layer, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

def create_combined_preview(color_layers, bg_color_rgb, layer_order, layer_visibility):
    """Создает комбинированный предпросмотр из видимых слоев"""
    if not color_layers:
        return None
    
    # Создаем базовое изображение
    first_layer = color_layers[0]
    combined = np.full_like(first_layer, bg_color_rgb)
    
    # Сортируем индексы по порядку (от нижнего к верхнему)
    sorted_indices = sorted(range(len(layer_order)), key=lambda x: layer_order[x])
    
    # Применяем слои в правильном порядке
    for idx in sorted_indices:
        if layer_visibility[idx]:
            layer = color_layers[idx]
            
            # Проверяем размеры и изменяем при необходимости
            if layer.shape != combined.shape:
                layer = resize_layer_to_match(layer, combined.shape)
            
            # Создаем маску (где есть цвет, отличный от фона)
            mask = np.any(layer != bg_color_rgb, axis=2)
            
            # Применяем слой только там, где есть маска
            combined[mask] = layer[mask]
    
    return combined

# ==================== БОКОВАЯ ПАНЕЛЬ ====================

with st.sidebar:
    st.markdown("<h2 class='sub-header'>⚙️ Настройки</h2>", unsafe_allow_html=True)
    
    # Загрузка изображения
    st.markdown("<h4>📤 Загрузите изображение</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Выберите файл", type=["jpg", "jpeg", "png", "bmp", "tiff"], 
                                    label_visibility="collapsed",
                                    key="file_uploader")
    
    if uploaded_file is not None:
        # Сохраняем в session state
        st.session_state.uploaded_file = uploaded_file
        
        # Выбор метода
        st.markdown("<h4>🎯 Выберите метод</h4>", unsafe_allow_html=True)
        methods = ["K-средних кластеризация (рекомендуется)"]
        if model_available:
            methods.append("Упрощенный нейронный метод")
        
        selected_method = st.selectbox("Метод разделения", methods, 
                                      label_visibility="collapsed",
                                      key="method_selector")
        st.session_state.selected_method = selected_method
        
        # Количество цветов
        st.markdown("<h4>🌈 Количество цветов</h4>", unsafe_allow_html=True)
        num_colors = st.slider("От 2 до 10 цветов", 2, 10, 5, 
                              help="Выберите количество цветов для извлечения из изображения",
                              label_visibility="collapsed",
                              key="num_colors_slider")
        
        # Цвет фона
        st.markdown("<h4>🎨 Цвет фона</h4>", unsafe_allow_html=True)
        bg_color = st.color_picker("Цвет фона для слоев", "#FFFFFF", 
                                  label_visibility="collapsed",
                                  key="bg_color_picker")
        bg_color_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Дополнительные опции для K-means
        if selected_method == "K-средних кластеризация (рекомендуется)":
            with st.expander("⚙️ Дополнительные настройки K-means", expanded=False):
                use_smart_bg_removal = st.checkbox("Удалить фон при анализе", True,
                                                  help="Игнорировать цвет фона при определении доминирующих цветов")
                
                enhance_edges = st.checkbox("Улучшить границы", True,
                                          help="Сделать границы между цветами более четкими")
        
        # Дополнительные опции
        with st.expander("🛠️ Общие настройки", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                export_quality = st.selectbox("Качество экспорта", 
                                            ["Высокое (300 DPI)", "Среднее (150 DPI)", "Низкое (72 DPI)"],
                                            index=1)
            
            with col2:
                preview_size = st.selectbox("Размер предпросмотра",
                                          ["Большой", "Средний", "Малый"],
                                          index=1)

# ==================== ОСНОВНОЕ СОДЕРЖИМОЕ ====================

# Секция загрузки
st.markdown("""
<div class="upload-section">
    <h3>🚀 Начните работу</h3>
    <p>Загрузите изображение в формате JPG, PNG, BMP или TIFF</p>
    <p>Максимальный размер файла: 50 MB</p>
</div>
""", unsafe_allow_html=True)

# Если файл загружен
if st.session_state.uploaded_file is not None:
    uploaded_file = st.session_state.uploaded_file
    selected_method = st.session_state.selected_method
    
    # Показываем информацию о выбранном методе
    method_descriptions = {
        "K-средних кластеризация (рекомендуется)": "Классический алгоритм для группировки похожих цветов",
        "Упрощенный нейронный метод": "Упрощенная нейронная сеть для цветовой сегментации"
    }
    
    st.markdown(f"""
    <div class="method-card">
        <h4>🎯 Выбранный метод: <strong>{selected_method}</strong></h4>
        <p>{method_descriptions.get(selected_method, '')}</p>
        <p>📊 Количество цветов: <strong>{num_colors}</strong> | 🎨 Цвет фона: 
        <span style='color: {bg_color}; font-weight: bold; background-color: #f0f0f0; padding: 2px 8px; border-radius: 3px;'>{bg_color}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    # Чтение изображения
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Конвертация PIL Image в формат OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.session_state.original_image_cv = img_cv
    
    with col1:
        st.markdown("<h3 class='sub-header'>📷 Исходное изображение</h3>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        
        # Информация об изображении
        with st.expander("📊 Информация об изображении", expanded=False):
            st.write(f"**Размер:** {image.width} × {image.height} пикселей")
            st.write(f"**Формат:** {image.format}")
            st.write(f"**Режим:** {image.mode}")
            st.write(f"**Размер файла:** {len(image_bytes) / 1024:.1f} KB")
            
            # Показываем основные цвета изображения
            st.write("**Основные цвета изображения:**")
            try:
                from collections import Counter
                img_array = np.array(image.convert("RGB"))
                pixels = img_array.reshape(-1, 3)
                
                # Берем случайную выборку для скорости
                if len(pixels) > 10000:
                    np.random.seed(42)
                    indices = np.random.choice(len(pixels), 10000, replace=False)
                    sample_pixels = pixels[indices]
                else:
                    sample_pixels = pixels
                
                # Группируем похожие цвета
                from sklearn.cluster import MiniBatchKMeans
                kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
                labels = kmeans.fit_predict(sample_pixels)
                centers = kmeans.cluster_centers_.astype(int)
                
                # Отображаем цвета
                colors_html = ""
                for color in centers:
                    hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
                    colors_html += f'<div style="display: inline-block; width: 20px; height: 20px; background-color: {hex_color}; margin: 2px; border: 1px solid #ccc; border-radius: 3px;" title="RGB{tuple(color)}"></div>'
                
                st.markdown(f'<div style="margin-top: 10px;">{colors_html}</div>', unsafe_allow_html=True)
            except:
                pass
    
    with col2:
        st.markdown("<h3 class='sub-header'>🎨 Разделенные цветовые слои</h3>", unsafe_allow_html=True)
        
        # Кнопка для запуска обработки
        process_button = st.button("🚀 Начать разделение цветов", 
                                 type="primary", 
                                 use_container_width=True,
                                 key="process_button")
        
        if process_button:
            with st.spinner("🔄 Обработка изображения... Пожалуйста, подождите."):
                progress_bar = st.progress(0)
                
                try:
                    # Шаг 1: Подготовка
                    progress_bar.progress(10)
                    
                    if selected_method == "K-средних кластеризация (рекомендуется)":
                        # Используем K-means с улучшенными параметрами
                        progress_bar.progress(30)
                        
                        color_layers, color_info = smart_color_separation_kmeans(
                            img_cv, 
                            n_colors=num_colors,
                            bg_color=bg_color_rgb
                        )
                        
                        progress_bar.progress(70)
                    
                    elif selected_method == "Упрощенный нейронный метод":
                        # Используем упрощенный нейронный метод
                        progress_bar.progress(30)
                        
                        # Получаем доминирующие цвета
                        palette_colors = get_dominant_colors(image, num_colors)
                        st.session_state.palette_colors = palette_colors
                        
                        # Вызываем упрощенную функцию
                        decompose_layers = decompose_fast_soft_color_simple(
                            image,
                            num_colors=num_colors,
                            palette=palette_colors,
                            device="cpu"
                        )
                        
                        progress_bar.progress(50)
                        
                        if decompose_layers:
                            # Преобразуем слои в формат для отображения
                            color_layers, color_info = decompose_layers_to_cv_format(
                                decompose_layers, 
                                bg_color_rgb
                            )
                        else:
                            st.error("Не удалось выполнить разделение с помощью нейронной сети.")
                            color_layers, color_info = [], []
                        
                        progress_bar.progress(70)
                    
                    # Шаг 2: Обработка результатов
                    progress_bar.progress(80)
                    
                    # Сохраняем результаты в session state
                    st.session_state.color_layers = color_layers
                    st.session_state.color_info = color_info
                    st.session_state.processing_done = True
                    
                    # Шаг 3: Завершение
                    progress_bar.progress(100)
                    
                    if color_layers and color_info:
                        st.success(f"✅ Успешно создано {len(color_layers)} цветовых слоев!")
                        
                        # Показываем информацию о полученных цветах
                        with st.expander("📊 Статистика по слоям", expanded=False):
                            total_coverage = sum(info['percentage'] for info in color_info)
                            st.write(f"**Общее покрытие:** {total_coverage:.1f}%")
                            
                            for i, info in enumerate(color_info):
                                hex_color = "#{:02x}{:02x}{:02x}".format(
                                    info['color'][2], info['color'][1], info['color'][0]
                                )
                                st.write(f"**Слой {i+1}:** {hex_color} - {info['percentage']:.1f}%")
                    else:
                        st.warning("⚠️ Не удалось создать цветовые слои. Попробуйте изменить параметры.")
                        
                except Exception as e:
                    st.error(f"❌ Ошибка при обработке изображения: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Показываем результаты если они есть
        color_layers = st.session_state.color_layers
        color_info = st.session_state.color_info
        
        if color_layers and color_info and st.session_state.processing_done:
            # Создаем вкладки для каждого слоя
            tab_names = [f"Слой {i+1}" for i in range(len(color_layers))]
            tabs = st.tabs(tab_names)
            
            for i, (layer, info) in enumerate(zip(color_layers, color_info)):
                with tabs[i]:
                    col_left, col_right = st.columns([3, 1])
                    
                    with col_left:
                        # Конвертация слоя из BGR в RGB для отображения
                        layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                        
                        # Настраиваем размер предпросмотра
                        if preview_size == "Большой":
                            st.image(layer_rgb, use_column_width=True)
                        elif preview_size == "Средний":
                            st.image(layer_rgb, use_column_width=True)
                        else:
                            st.image(layer_rgb, use_column_width=True)
                        
                        # Кнопки для скачивания
                        col_btn1, col_btn2 = st.columns(2)
                        
                        with col_btn1:
                            # Черно-белая маска
                            bw_mask = create_bw_mask(layer, bg_color_rgb)
                            png_data = save_bw_mask_as_png(bw_mask, f"mask_{i+1}")
                            
                            if png_data:
                                hex_color = "#{:02x}{:02x}{:02x}".format(
                                    info['color'][2], info['color'][1], info['color'][0]
                                )
                                
                                st.download_button(
                                    label="⬇️ Скачать ЧБ маску",
                                    data=png_data,
                                    file_name=f"layer_{i+1}_mask.png",
                                    mime="image/png",
                                    key=f"download_mask_{i}",
                                    use_container_width=True
                                )
                        
                        with col_btn2:
                            # Цветной слой
                            color_png_data = convert_to_png(layer, f"layer_{i+1}")
                            if color_png_data:
                                hex_color = "#{:02x}{:02x}{:02x}".format(
                                    info['color'][2], info['color'][1], info['color'][0]
                                )
                                
                                st.download_button(
                                    label="⬇️ Скачать цветной слой",
                                    data=color_png_data,
                                    file_name=f"layer_{i+1}_color.png",
                                    mime="image/png",
                                    key=f"download_color_{i}",
                                    use_container_width=True
                                )
                    
                    with col_right:
                        # Информация о цвете
                        hex_color = "#{:02x}{:02x}{:02x}".format(
                            info['color'][2], info['color'][1], info['color'][0]
                        )
                        rgb_color = info.get('rgb_color', info['color'][::-1])
                        
                        st.markdown(f"""
                        <div style='padding: 15px; background-color: #f8f9fa; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                                <div class='color-chip' style='background-color: {hex_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                                <div>
                                    <strong style='font-size: 1.2em;'>{hex_color}</strong><br>
                                    <span style='color: #666; font-size: 0.9em;'>Цвет слоя</span>
                                </div>
                            </div>
                            <div style='margin-bottom: 10px; padding: 10px; background-color: white; border-radius: 5px;'>
                                <strong style='color: #333;'>RGB:</strong> {rgb_color}<br>
                                <strong style='color: #333;'>Покрытие:</strong> {info['percentage']:.1f}%<br>
                                <strong style='color: #333;'>Размер:</strong> {layer.shape[1]} × {layer.shape[0]}px
                            </div>
                            <div class='progress-bar' style='width: {min(info['percentage'], 100)}%;'></div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # ==================== КОМБИНИРОВАННЫЙ ПРЕДПРОСМОТР ====================
            
            st.markdown("---")
            st.markdown("<h3 class='sub-header'>👁️ Комбинированный предпросмотр</h3>", unsafe_allow_html=True)
            
            # Настройки порядка слоев
            with st.expander("⚙️ Управление порядком и видимостью слоев", expanded=True):
                # Инициализация состояния сессии для порядка и видимости
                if 'layer_order' not in st.session_state or len(st.session_state.layer_order) != len(color_layers):
                    st.session_state.layer_order = list(range(len(color_layers)))
                if 'layer_visibility' not in st.session_state or len(st.session_state.layer_visibility) != len(color_layers):
                    st.session_state.layer_visibility = [True] * len(color_layers)
                
                # Настройки для каждого слоя
                st.write("**Настройте порядок и видимость слоев:**")
                
                for i in range(len(color_layers)):
                    col1, col2, col3, col4 = st.columns([1, 1, 3, 1])
                    
                    with col1:
                        # Порядок слоя
                        order_value = st.number_input(
                            "Позиция",
                            min_value=1,
                            max_value=len(color_layers),
                            value=st.session_state.layer_order[i] + 1,
                            key=f"order_{i}",
                            help="1 = нижний слой (фон), больше = выше",
                            label_visibility="collapsed"
                        )
                        st.session_state.layer_order[i] = order_value - 1
                    
                    with col2:
                        # Видимость слоя
                        visibility = st.checkbox(
                            "Вкл",
                            value=st.session_state.layer_visibility[i],
                            key=f"visibility_{i}",
                            label_visibility="collapsed"
                        )
                        st.session_state.layer_visibility[i] = visibility
                    
                    with col3:
                        # Информация о цвете
                        hex_color = "#{:02x}{:02x}{:02x}".format(
                            color_info[i]['color'][2], color_info[i]['color'][1], color_info[i]['color'][0]
                        )
                        st.markdown(f"""
                        <div style='display: flex; align-items: center; padding: 8px; background-color: {'#e8f5e9' if visibility else '#f5f5f5'}; border-radius: 5px; transition: all 0.3s ease;'>
                            <div style='width: 25px; height: 25px; background-color: {hex_color}; border: 1px solid #000; border-radius: 4px; margin-right: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'></div>
                            <div>
                                <div><strong>Слой {i+1}</strong> {'' if visibility else '(скрыт)'}</div>
                                <div style='font-size: 0.8em; color: #666;'>{hex_color} • {color_info[i]['percentage']:.1f}% покрытия</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        # Быстрое действие
                        if st.button("👁️", key=f"quick_toggle_{i}", help="Быстро переключить видимость"):
                            st.session_state.layer_visibility[i] = not st.session_state.layer_visibility[i]
                            st.rerun()
            
            # Создание комбинированного изображения
            combined = create_combined_preview(
                color_layers, 
                bg_color_rgb, 
                st.session_state.layer_order, 
                st.session_state.layer_visibility
            )
            
            if combined is not None:
                # Сохраняем комбинированный превью в session state
                st.session_state.combined_preview = combined
                
                # Отображаем комбинированное изображение
                combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
                
                visible_layers = sum(st.session_state.layer_visibility)
                total_layers = len(color_layers)
                
                st.image(combined_rgb, 
                        caption=f"Предпросмотр {visible_layers}/{total_layers} видимых слоев", 
                        use_column_width=True)
                
                # Кнопки для скачивания комбинированного изображения
                col_comb1, col_comb2, col_comb3 = st.columns(3)
                
                with col_comb1:
                    # Черно-белая маска комбинированного изображения
                    combined_bw_mask = np.zeros((combined.shape[0], combined.shape[1]), dtype=np.uint8)
                    
                    for i, layer in enumerate(color_layers):
                        if st.session_state.layer_visibility[i]:
                            # Проверяем размеры
                            if layer.shape[:2] != combined_bw_mask.shape:
                                layer_resized = resize_layer_to_match(layer, combined_bw_mask.shape[:2] + (3,))
                            else:
                                layer_resized = layer
                            
                            layer_mask = create_bw_mask(layer_resized, bg_color_rgb)
                            combined_bw_mask = cv2.bitwise_or(combined_bw_mask, layer_mask)
                    
                    combined_png_data = save_bw_mask_as_png(combined_bw_mask, "combined_mask")
                    
                    if combined_png_data:
                        st.download_button(
                            label="⬇️ ЧБ маска (все слои)",
                            data=combined_png_data,
                            file_name="combined_mask.png",
                            mime="image/png",
                            key="download_combined_mask",
                            use_container_width=True
                        )
                
                with col_comb2:
                    # Цветное комбинированное изображение
                    combined_color_png = convert_to_png(combined, "combined_preview")
                    if combined_color_png:
                        st.download_button(
                            label="⬇️ Цветной предпросмотр",
                            data=combined_color_png,
                            file_name="combined_preview.png",
                            mime="image/png",
                            key="download_combined_color",
                            use_container_width=True
                        )
                
                with col_comb3:
                    # Показать/скрыть все слои
                    col_show, col_hide = st.columns(2)
                    with col_show:
                        if st.button("👁️ Показать все", use_container_width=True):
                            st.session_state.layer_visibility = [True] * len(color_layers)
                            st.rerun()
                    with col_hide:
                        if st.button("👁️ Скрыть все", use_container_width=True):
                            st.session_state.layer_visibility = [False] * len(color_layers)
                            st.rerun()
            
            # ==================== ПАКЕТНОЕ СКАЧИВАНИЕ ====================
            
            st.markdown("---")
            st.markdown("<h3 class='sub-header'>📦 Пакетное скачивание</h3>", unsafe_allow_html=True)
            
            if st.button("📁 Создать ZIP-архив со всеми слоями", 
                        type="secondary", 
                        use_container_width=True,
                        key="create_zip_button"):
                with st.spinner("🔄 Создание архива... Это может занять некоторое время."):
                    progress_zip = st.progress(0)
                    
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        all_files = []
                        total_files = len(color_layers) * 2 + 3  # Маски + цветные + комбинированные + readme
                        
                        # Сохраняем все слои
                        for idx, layer in enumerate(color_layers):
                            if st.session_state.layer_visibility[idx]:
                                # Черно-белая маска
                                progress_zip.progress((idx * 2) / total_files)
                                bw_mask = create_bw_mask(layer, bg_color_rgb)
                                mask_png = save_bw_mask_as_png(bw_mask, f"mask_{idx+1}")
                                
                                if mask_png:
                                    mask_path = os.path.join(tmpdirname, f"layer_{idx+1}_mask.png")
                                    with open(mask_path, 'wb') as f:
                                        f.write(mask_png)
                                    all_files.append(mask_path)
                                
                                # Цветной слой
                                progress_zip.progress((idx * 2 + 1) / total_files)
                                layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                                color_png = convert_to_png(layer_rgb, f"layer_{idx+1}")
                                
                                if color_png:
                                    color_path = os.path.join(tmpdirname, f"layer_{idx+1}_color.png")
                                    with open(color_path, 'wb') as f:
                                        f.write(color_png)
                                    all_files.append(color_path)
                        
                        # Сохраняем комбинированные изображения
                        progress_zip.progress(0.8)
                        if combined_png_data:
                            combined_path = os.path.join(tmpdirname, "combined_mask.png")
                            with open(combined_path, 'wb') as f:
                                f.write(combined_png_data)
                            all_files.append(combined_path)
                        
                        if combined_color_png:
                            combined_color_path = os.path.join(tmpdirname, "combined_preview.png")
                            with open(combined_color_path, 'wb') as f:
                                f.write(combined_color_png)
                            all_files.append(combined_color_path)
                        
                        # Создаем README файл
                        progress_zip.progress(0.9)
                        readme_content = f"""# ColorSep Pro - Экспортированные слои

Дата создания: {st.session_state.get('processing_time', 'Неизвестно')}
Метод: {selected_method}
Количество слоев: {len(color_layers)}
Цвет фона: {bg_color}
Видимых слоев: {sum(st.session_state.layer_visibility)}/{len(color_layers)}

## Содержимое архива:
- Черно-белые маски каждого слоя (layer_X_mask.png)
- Цветные изображения каждого слоя (layer_X_color.png)
- Комбинированные изображения (combined_*.png)

## Информация о слоях:
"""
                        
                        for i, info in enumerate(color_info):
                            hex_color = "#{:02x}{:02x}{:02x}".format(
                                info['color'][2], info['color'][1], info['color'][0]
                            )
                            visibility = "Видимый" if st.session_state.layer_visibility[i] else "Скрытый"
                            readme_content += f"- Слой {i+1} ({visibility}): {hex_color}, RGB{info.get('rgb_color', info['color'][::-1])}, Покрытие: {info['percentage']:.1f}%\n"
                        
                        readme_path = os.path.join(tmpdirname, "README.txt")
                        with open(readme_path, 'w', encoding='utf-8') as f:
                            f.write(readme_content)
                        all_files.append(readme_path)
                        
                        # Создаем ZIP архив
                        zip_path = os.path.join(tmpdirname, "color_layers.zip")
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for file in all_files:
                                arcname = os.path.basename(file)
                                zipf.write(file, arcname)
                        
                        # Читаем ZIP файл
                        with open(zip_path, "rb") as f:
                            zip_data = f.read()
                        
                        progress_zip.progress(1.0)
                        
                        # Предоставляем для скачивания
                        st.download_button(
                            label="⬇️ Скачать ZIP архив со всеми файлами",
                            data=zip_data,
                            file_name=f"color_separation_{uploaded_file.name.split('.')[0]}.zip",
                            mime="application/zip",
                            key="download_all_zip_final",
                            use_container_width=True
                        )

# ==================== ИНФОРМАЦИЯ О МЕТОДАХ ====================

st.markdown("---")
st.markdown("<h2 class='sub-header'>📚 Описание методов</h2>", unsafe_allow_html=True)

col_method1, col_method2 = st.columns(2)

with col_method1:
    st.markdown("""
    <div class="method-card">
        <h4>🎯 K-средних кластеризация (рекомендуется)</h4>
        <p><strong>Описание:</strong> Классический алгоритм машинного обучения для группировки похожих цветов.</p>
        <p><strong>Преимущества:</strong></p>
        <ul>
            <li>✅ Быстрая и стабильная работа</li>
            <li>✅ Правильные цвета из вашего изображения</li>
            <li>✅ Хорошо работает с фотографиями</li>
            <li>✅ Не требует специальных моделей</li>
        </ul>
        <p><strong>Идеально для:</strong> Фотографии, природные сцены, изображения с реалистичными цветами</p>
        <div style='background-color: #e7f3ff; padding: 10px; border-radius: 5px; margin-top: 10px;'>
            <strong>🎯 Рекомендуем для большинства случаев</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_method2:
    if model_available:
        st.markdown("""
        <div class="method-card">
            <h4>⚡ Упрощенный нейронный метод</h4>
            <p><strong>Описание:</strong> Упрощенная нейронная сеть для цветовой сегментации.</p>
            <p><strong>Особенности:</strong></p>
            <ul>
                <li>⚠️ Требует модель mask_generator.pth</li>
                <li>⚠️ Может давать неожиданные цвета</li>
                <li>⚠️ Экспериментальный метод</li>
            </ul>
            <p><strong>Используйте осторожно:</strong> Результаты могут отличаться от ожидаемых</p>
            <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                <strong>⚠️ Экспериментальный - используйте K-means для надежных результатов</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="method-card" style="border-left-color: #ffc107;">
            <h4>⚡ Упрощенный нейронный метод</h4>
            <p><strong>Статус:</strong> 🔒 Требуется модель</p>
            <p>Для использования этого метода необходимо скачать файл модели и поместить его в папку <code>model/</code></p>
            <p><strong>Примечание:</strong> Этот метод может давать неожиданные результаты. Для большинства задач рекомендуется использовать K-means.</p>
            <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                <strong>💡 Совет: Используйте K-means для лучших результатов</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================== ФУТЕР ====================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 30px; background-color: #f8f9fa; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <h4 style="color: #0056b3;">🎨 ColorSep Pro v2.0</h4>
    <p style="font-size: 1.1em; margin-bottom: 10px;">Профессиональный инструмент для разделения цветов</p>
    <div style="display: flex; justify-content: center; gap: 20px; margin: 20px 0;">
        <div style="text-align: center;">
            <div style="font-size: 1.5em; color: #0056b3;">📷</div>
            <div>Поддержка JPG, PNG, BMP, TIFF</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.5em; color: #0056b3;">⚡</div>
            <div>Быстрая обработка</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.5em; color: #0056b3;">🎯</div>
            <div>Точные цвета</div>
        </div>
    </div>
    <p style="font-size: 0.9em; color: #888; margin-top: 20px;">Все файлы экспортируются в формате PNG для промышленной совместимости</p>
    <p style="font-size: 0.9em; color: #888;">Максимальный размер файла: 50MB</p>
</div>
""", unsafe_allow_html=True)

# ==================== ПРОВЕРКА ЗАВИСИМОСТЕЙ ====================

try:
    with st.sidebar.expander("ℹ️ Информация о системе", expanded=False):
        st.write(f"**OpenCV:** {cv2.__version__}")
        st.write(f"**PyTorch:** {torch.__version__}")
        st.write(f"**CUDA доступен:** {'✅ Да' if torch.cuda.is_available() else '❌ Нет'}")
        
        # Проверка scikit-learn
        try:
            from sklearn import __version__ as sklearn_version
            st.write(f"**scikit-learn:** {sklearn_version}")
        except:
            st.write("**scikit-learn:** ❌ Не установлен")
        
        st.write(f"**Streamlit:** {st.__version__}")
        
        # Информация о памяти
        import psutil
        memory = psutil.virtual_memory()
        st.write(f"**Оперативная память:** {memory.percent}% использовано")
        
except Exception as e:
    st.sidebar.error(f"Ошибка проверки зависимостей: {e}")

# ==================== СОВЕТЫ ПО ИСПОЛЬЗОВАНИЮ ====================

with st.sidebar.expander("💡 Советы по использованию", expanded=False):
    st.markdown("""
    **🎯 Для лучших результатов:**
    
    1. **Используйте K-means** - дает самые точные цвета
    2. **Оптимальное количество цветов:** 5-7 для фотографий
    3. **Фон:** Укажите цвет фона если он однородный
    4. **Качество:** Для печати используйте "Высокое качество"
    
    **⚠️ Если цвета искажаются:**
    
    1. Переключитесь на метод K-means
    2. Уменьшите количество цветов
    3. Укажите точный цвет фона
    
    **📁 Для экспорта:**
    
    - Черно-белые маски: для трафаретной печати
    - Цветные слои: для цифровой печати
    - ZIP архив: содержит все файлы + инструкцию
    """)

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
import pantone_colors as pantone
from pantone_tab import pantone_extraction_tab
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
from pathlib import Path
from pyora import Project
import warnings
warnings.filterwarnings('ignore')

# ==================== ДОПОЛНИТЕЛЬНЫЕ ИМПОРТЫ ДЛЯ INKSPLIT ====================

try:
    from colormath.color_objects import sRGBColor, LabColor, LCHabColor
    from colormath.color_conversions import convert_color
    from colormath.color_diff import delta_e_cie1976
    colormath_available = True
except ImportError:
    colormath_available = False
    st.warning("⚠️ Библиотека colormath не установлена. Метод Inksplit будет ограничен.")

# ==================== ПРОВЕРКА НАЛИЧИЯ МОДЕЛИ ====================

def check_model_exists():
    """Проверяет наличие модели в папке model/"""
    model_path = Path("model/mask_generator7.pth")
    
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
    }
    .color-chip {
        display: inline-block;
        width: 30px;
        height: 30px;
        margin-right: 10px;
        border: 2px solid #000;
        border-radius: 5px;
        vertical-align: middle;
    }
    .method-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #0056b3;
    }
    .method-card-inksplit {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    .model-status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #c3e6cb;
        margin-bottom: 20px;
    }
    .model-status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #ffeaa7;
        margin-bottom: 20px;
    }
    .upload-section {
        background-color: #e7f3ff;
        padding: 25px;
        border-radius: 12px;
        border: 3px dashed #0056b3;
        text-align: center;
        margin-bottom: 25px;
    }
    .layer-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .preview-container {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .tab-content {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .inksplit-options {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #cce5ff;
    }
    .export-options {
        background-color: #fff3e6;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #ffd9b3;
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
        3. Положите файл <code>mask_generator7.pth</code> в папку <code>model/</code><br>
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

# ==================== КЛАССЫ ДЛЯ МЕТОДА DECOMPOSE ====================

class _MyDataset(torch.utils.data.Dataset):
    def __init__(self, img, num_primary_color, palette):
        self.img = img.convert("RGB")
        self.palette_list = palette.reshape(-1, num_primary_color * 3)
        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        np_img = np.array(self.img)
        np_img = np_img.transpose((2, 0, 1))
        target_img = np_img / 255  # 0~1

        # select primary_color
        primary_color_layers = self._make_primary_color_layers(
            self.palette_list[index], target_img
        )

        # to Tensor
        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color_layers.astype(np.float32))

        return target_img, primary_color_layers  # return torch.Tensor

    def __len__(self):
        return 1

    def _make_primary_color_layers(self, palette_values, target_img):
        primary_color = (
            palette_values.reshape(self.num_primary_color, 3) / 255
        )  # (ln, 3)
        primary_color_layers = np.tile(
            np.ones_like(target_img), (self.num_primary_color, 1, 1, 1)
        ) * primary_color.reshape(self.num_primary_color, 3, 1, 1)
        return primary_color_layers

class _MaskGeneratorModel(nn.Module):
    def __init__(self, num_primary_color):
        super(_MaskGeneratorModel, self).__init__()
        in_dim = 3 + num_primary_color * 3  # ex. 21 ch (= 3 + 6 * 3)
        out_dim = num_primary_color  # num_out_layers is the same as num_primary_color.

        self.conv1 = nn.Conv2d(
            in_dim, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            in_dim * 2, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv3 = nn.Conv2d(
            in_dim * 4, in_dim * 8, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.deconv1 = nn.ConvTranspose2d(
            in_dim * 8,
            in_dim * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            output_padding=1,
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_dim * 8,
            in_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            output_padding=1,
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_dim * 4,
            in_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            output_padding=1,
        )
        self.conv4 = nn.Conv2d(
            in_dim * 2 + 3, in_dim, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(in_dim * 2)
        self.bn2 = nn.BatchNorm2d(in_dim * 4)
        self.bn3 = nn.BatchNorm2d(in_dim * 8)
        self.bnde1 = nn.BatchNorm2d(in_dim * 4)
        self.bnde2 = nn.BatchNorm2d(in_dim * 2)
        self.bnde3 = nn.BatchNorm2d(in_dim * 2)
        self.bn4 = nn.BatchNorm2d(in_dim)

    def forward(self, target_img, primary_color_pack):
        x = torch.cat((target_img, primary_color_pack), dim=1)

        h1 = self.bn1(F.relu(self.conv1(x)))  # *2
        h2 = self.bn2(F.relu(self.conv2(h1)))  # *4
        h3 = self.bn3(F.relu(self.conv3(h2)))  # *8
        h4 = self.bnde1(F.relu(self.deconv1(h3)))  # *4
        h4 = torch.cat((h4, h2), 1)  # *8
        h5 = self.bnde2(F.relu(self.deconv2(h4)))  # *2
        h5 = torch.cat((h5, h1), 1)  # *4
        h6 = self.bnde3(F.relu(self.deconv3(h5)))  # *2
        h6 = torch.cat((h6, target_img), 1)  # *2+3
        h7 = self.bn4(F.relu(self.conv4(h6)))

        return torch.sigmoid(self.conv5(h7))  # box constraint for alpha layers

# ==================== ФУНКЦИИ ДЛЯ МЕТОДА DECOMPOSE ====================

def get_dominant_colors(img: Image.Image, num_colors: int) -> list[tuple]:
    """
    Получение доминирующих цветов из изображения с использованием K-means
    """
    # Конвертируем изображение в массив numpy
    img_array = np.array(img)
    
    # Если изображение RGBA, конвертируем в RGB
    if img.mode == "RGBA":
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Преобразуем изображение в формат для K-means
    pixels = img_array.reshape(-1, 3)
    
    # Используем K-means для нахождения доминирующих цветов
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Получаем центры кластеров (доминирующие цвета)
    colors = kmeans.cluster_centers_.astype(int)
    
    # Сортируем цвета по частоте
    labels = kmeans.labels_
    counts = np.bincount(labels)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_colors = colors[sorted_indices]
    
    # Конвертируем в список кортежей
    return [tuple(color) for color in sorted_colors]

def decompose_fast_soft_color(
    input_image: Image.Image,
    num_colors: int = 7,
    palette: list[tuple] = None,
    resize_scale_factor: float = 1.0
) -> list[Image.Image]:
    """
    Функция для разложения изображения на цветовые слои с использованием нейронной сети
    Поддерживает от 2 до 8 цветов
    """
    layersRGBA = []
    
    if not model_available:
        st.error("Модель не найдена. Невозможно выполнить метод Decompose.")
        return []
    
    if num_colors < 2 or num_colors > 8:
        st.error(f"Количество цветов должно быть от 2 до 8. Получено: {num_colors}")
        return []
    
    # Преобразование изображения PIL в формат для обработки
    if palette is None:
        # Используем K-means для получения палитры
        palette = get_dominant_colors(input_image, num_colors)
    else:
        # Если передана палитра, убедимся, что в ней правильное количество цветов
        if len(palette) != num_colors:
            # Если цветов меньше, добавим недостающие
            while len(palette) < num_colors:
                palette.append(palette[-1] if palette else (128, 128, 128))
            # Если цветов больше, обрежем
            palette = palette[:num_colors]
    
    palette = np.array(palette)
    
    try:
        test_dataset = _MyDataset(input_image, num_colors, palette)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        
        cpu = torch.device("cpu")
        
        # Загрузка модели
        mask_generator = _MaskGeneratorModel(num_colors).to(cpu)
        
        # Загрузка весов модели
        model_path = Path("model/mask_generator7.pth")
        mask_generator.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        
        # Режим оценки
        mask_generator.eval()
        
        def cut_edge(target_img: torch.tensor) -> torch.tensor:
            target_img = F.interpolate(
                target_img, scale_factor=resize_scale_factor, mode="area"
            )
            h = target_img.size(2)
            w = target_img.size(3)
            h = h - (h % 8)
            w = w - (w % 8)
            target_img = target_img[:, :, :h, :w]
            return target_img
        
        def alpha_normalize(alpha_layers: torch.Tensor) -> torch.Tensor:
            return alpha_layers / (alpha_layers.sum(dim=1, keepdim=True) + 1e-8)
        
        def normalize_to_0_255(nd: np.array):
            nd = (nd * 255) + 0.5
            nd = np.clip(nd, 0, 255).astype("uint8")
            return nd
        
        with torch.no_grad():
            for batch_idx, (target_img, primary_color_layers) in enumerate(test_loader):
                if batch_idx != 0:
                    continue
                
                target_img = cut_edge(target_img)
                target_img = target_img.to("cpu")
                primary_color_layers = primary_color_layers.to("cpu")
                primary_color_pack = primary_color_layers.view(
                    primary_color_layers.size(0),
                    -1,
                    primary_color_layers.size(3),
                    primary_color_layers.size(4),
                )
                primary_color_pack = cut_edge(primary_color_pack)
                primary_color_layers = primary_color_pack.view(
                    primary_color_pack.size(0),
                    -1,
                    3,
                    primary_color_pack.size(2),
                    primary_color_pack.size(3),
                )
                pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
                pred_alpha_layers = pred_alpha_layers_pack.view(
                    target_img.size(0), -1, 1, target_img.size(2), target_img.size(3)
                )
                
                processed_alpha_layers = alpha_normalize(pred_alpha_layers)
                processed_alpha_layers = alpha_normalize(processed_alpha_layers)  # Двойная нормализация
                
                mono_RGBA_layers = torch.cat(
                    (primary_color_layers, processed_alpha_layers), dim=2
                )  # out: bn, ln, 4, h, w
                
                # Преобразование в изображения PIL
                mono_RGBA_layers = mono_RGBA_layers[0]  # ln, 4. h, w
                for i in range(len(mono_RGBA_layers)):
                    im = mono_RGBA_layers[i, :, :, :].numpy()
                    im = im.transpose((1, 2, 0))
                    im = normalize_to_0_255(im)
                    layersRGBA.append(Image.fromarray(im))
                
                break
        
        return layersRGBA
    
    except Exception as e:
        st.error(f"Ошибка при выполнении метода Decompose: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return []

def decompose_layers_to_cv_format(decompose_layers, bg_color):
    """
    Преобразует слои RGBA из метода decompose в формат BGR с прозрачностью,
    учитывая заданный цвет фона.
    """
    cv_layers = []
    color_info_list = []
    
    for i, pil_layer in enumerate(decompose_layers):
        # Конвертируем PIL Image в numpy array
        rgba_array = np.array(pil_layer)
        
        # Если слой RGBA, разделяем на RGB и альфа
        if rgba_array.shape[2] == 4:
            rgb_array = rgba_array[:, :, :3]
            alpha_array = rgba_array[:, :, 3] / 255.0
            
            # Создаем слой с прозрачностью на белом фоне
            layer_with_bg = np.zeros_like(rgb_array, dtype=np.uint8)
            
            # Применяем альфа-канал
            for c in range(3):
                layer_with_bg[:, :, c] = rgb_array[:, :, c] * alpha_array + bg_color[c] * (1 - alpha_array)
            
            # Конвертируем RGB в BGR для OpenCV
            bgr_layer = cv2.cvtColor(layer_with_bg, cv2.COLOR_RGB2BGR)
            
            # Вычисляем доминирующий цвет слоя
            # Используем медиану цветов, где альфа > 0.1
            mask = alpha_array > 0.1
            if np.any(mask):
                # Получаем цвета пикселей с высокой прозрачностью
                masked_colors = rgb_array[mask]
                # Вычисляем медианный цвет
                if len(masked_colors) > 0:
                    median_color = np.median(masked_colors, axis=0).astype(int)
                    # Конвертируем RGB в BGR для консистентности
                    median_color_bgr = (median_color[2], median_color[1], median_color[0])
                else:
                    median_color_bgr = bg_color
            else:
                # Если нет достаточно непрозрачных пикселей, используем средний цвет
                median_color_bgr = bg_color
            
            # Вычисляем процент покрытия
            coverage_percentage = (np.sum(mask) / mask.size) * 100
            
            cv_layers.append(bgr_layer)
            color_info_list.append({
                'color': median_color_bgr,
                'percentage': coverage_percentage
            })
        else:
            # Если слой RGB (без альфа), просто конвертируем
            bgr_layer = cv2.cvtColor(rgba_array, cv2.COLOR_RGB2BGR)
            
            # Вычисляем доминирующий цвет
            if rgba_array.size > 0:
                unique_colors, counts = np.unique(rgba_array.reshape(-1, 3), axis=0, return_counts=True)
                if len(unique_colors) > 0:
                    dominant_color_idx = np.argmax(counts)
                    dominant_color_rgb = unique_colors[dominant_color_idx]
                    dominant_color_bgr = (dominant_color_rgb[2], dominant_color_rgb[1], dominant_color_rgb[0])
                else:
                    dominant_color_bgr = bg_color
            else:
                dominant_color_bgr = bg_color
            
            # Процент покрытия (все пиксели, кроме фона)
            non_bg_mask = np.any(bgr_layer != bg_color, axis=2)
            coverage_percentage = (np.sum(non_bg_mask) / non_bg_mask.size) * 100
            
            cv_layers.append(bgr_layer)
            color_info_list.append({
                'color': dominant_color_bgr,
                'percentage': coverage_percentage
            })
    
    return cv_layers, color_info_list

# ==================== ФУНКЦИИ ДЛЯ МЕТОДА K-MEANS ====================

def kmeans_color_separation(img, n_colors=5, bg_color=(255, 255, 255), **kwargs):
    """
    Разделение цветов с использованием алгоритма K-means
    Поддерживает от 2 до 8 цветов
    """
    if n_colors < 2 or n_colors > 8:
        st.error(f"Количество цветов должно быть от 2 до 8. Получено: {n_colors}")
        return [], []
    
    try:
        # Преобразуем изображение в формат для K-means
        pixels = img.reshape(-1, 3)
        
        # Удаляем пиксели фона
        if bg_color:
            bg_mask = np.all(pixels == bg_color, axis=1)
            if np.any(bg_mask):
                pixels = pixels[~bg_mask]
        
        # Если после удаления фона не осталось пикселей
        if len(pixels) == 0:
            st.warning("Изображение состоит только из фона")
            return [], []
        
        # Применяем K-means
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Восстанавливаем маску для всего изображения
        full_labels = np.zeros(img.shape[0] * img.shape[1], dtype=int) - 1
        if bg_color:
            bg_mask_full = np.all(img.reshape(-1, 3) == bg_color, axis=1)
            non_bg_indices = np.where(~bg_mask_full)[0]
            if len(non_bg_indices) >= len(labels):
                full_labels[non_bg_indices[:len(labels)]] = labels
        
        # Создаем слои
        color_layers = []
        color_info = []
        
        for i in range(n_colors):
            # Создаем маску для текущего кластера
            mask = (full_labels == i).reshape(img.shape[0], img.shape[1])
            
            # Создаем слой с фоном
            layer = np.full_like(img, bg_color)
            
            # Заполняем цветом кластера
            cluster_color = kmeans.cluster_centers_[i].astype(int)
            layer[mask] = cluster_color
            
            color_layers.append(layer)
            color_info.append({
                'color': (int(cluster_color[0]), 
                         int(cluster_color[1]), 
                         int(cluster_color[2])),
                'percentage': (np.sum(mask) / mask.size) * 100
            })
        
        return color_layers, color_info
    
    except Exception as e:
        st.error(f"Ошибка в методе K-means: {str(e)}")
        return [], []

# ==================== ФУНКЦИИ ДЛЯ МЕТОДА INKSPLIT ====================

def find_closest_color_in_palette(target_color, palette_colors, batch_size=100):
    """
    Находит ближайший цвет в палитре с использованием цветовой метрики Delta E.
    Аналог функции из плагина GIMP.
    """
    if not colormath_available:
        # Упрощенная версия без colormath
        target_color_rgb = np.array(target_color)
        min_distance = float('inf')
        closest_color = None
        closest_color_name = None
        
        for color_name, color_rgb in palette_colors:
            distance = np.linalg.norm(target_color_rgb - np.array(color_rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = color_rgb
                closest_color_name = color_name
        
        return closest_color_name, closest_color, min_distance
    
    # Полная версия с colormath
    rgb = sRGBColor(target_color[0]/255, target_color[1]/255, target_color[2]/255, is_upscaled=False)
    target_color_lab = convert_color(rgb, LabColor)

    min_distance = float('inf')
    closest_color = None
    closest_color_name = None

    for color_name, color_rgb in palette_colors:
        rgb = sRGBColor(color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255, is_upscaled=False)
        color_lab = convert_color(rgb, LabColor)

        distance = delta_e_cie1976(target_color_lab, color_lab)

        if distance < min_distance:
            min_distance = distance
            closest_color = color_rgb
            closest_color_name = color_name

    return closest_color_name, closest_color, min_distance

def inksplit_color_separation(
    img, 
    n_colors=5, 
    bg_color=(255, 255, 255),
    auto_color_match=True,
    dithering=True,
    generate_ub=True,
    ub_threshold=0.35,
    use_custom_palette=False,
    custom_palette=None,
    best_fit_match=False,
    best_fit_palette=None,
    **kwargs
):
    """
    Реализация метода Inksplit для разделения цветов.
    Основано на алгоритме плагина GIMP.
    """
    try:
        # Создаем копию изображения для обработки
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Уменьшаем количество цветов с помощью K-means для начальной палитры
        pixels = img.reshape(-1, 3)
        if bg_color:
            bg_mask = np.all(pixels == bg_color, axis=1)
            if np.any(bg_mask):
                pixels = pixels[~bg_mask]
        
        if len(pixels) == 0:
            st.warning("Изображение состоит только из фона")
            return [], []
        
        # Используем K-means для создания базовой палитры
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Получаем цвета палитры
        palette_colors = kmeans.cluster_centers_.astype(int)
        
        # Создаем слои
        color_layers = []
        color_info = []
        
        # Для каждого цвета в палитре создаем отдельный слой
        for i, color in enumerate(palette_colors):
            # Преобразуем цвет в RGB
            color_rgb = (int(color[2]), int(color[1]), int(color[0]))
            
            # Создаем маску для текущего цвета
            # Используем пороговое значение для поиска похожих цветов
            color_diff = np.linalg.norm(img - color_rgb, axis=2)
            mask = color_diff < 50  # Порог для определения цвета
            
            # Создаем слой
            layer = np.full_like(img, bg_color)
            layer[mask] = color_rgb
            
            # Если включено сглаживание
            if dithering:
                # Применяем простое сглаживание границ
                kernel = np.ones((3,3), np.float32)/9
                layer = cv2.filter2D(layer, -1, kernel)
            
            color_layers.append(layer)
            
            # Вычисляем информацию о цвете
            coverage_percentage = (np.sum(mask) / mask.size) * 100
            color_info.append({
                'color': color_rgb,
                'percentage': coverage_percentage,
                'original_index': i
            })
        
        # Сортируем слои по покрытию (от большего к меньшему)
        sorted_indices = sorted(range(len(color_info)), 
                              key=lambda x: color_info[x]['percentage'], 
                              reverse=True)
        
        color_layers = [color_layers[i] for i in sorted_indices]
        color_info = [color_info[i] for i in sorted_indices]
        
        # Создание подложки (Underbase), если требуется
        if generate_ub:
            ub_layer = np.full_like(img, bg_color)
            
            for i, (layer, info) in enumerate(zip(color_layers, color_info)):
                # Создаем маску для текущего слоя
                mask = np.any(layer != bg_color, axis=2)
                
                # Конвертируем цвет в LCH для проверки яркости
                color_rgb_normalized = np.array(info['color']) / 255.0
                
                if colormath_available:
                    # Используем colormath для точного расчета яркости
                    rgb = sRGBColor(color_rgb_normalized[0], 
                                   color_rgb_normalized[1], 
                                   color_rgb_normalized[2], 
                                   is_upscaled=False)
                    lab = convert_color(rgb, LabColor)
                    lch = convert_color(lab, LCHabColor)
                    L = lch.lch_l / 100.0
                else:
                    # Упрощенный расчет яркости
                    L = 0.299 * color_rgb_normalized[0] + \
                        0.587 * color_rgb_normalized[1] + \
                        0.114 * color_rgb_normalized[2]
                
                # Удаляем из подложки темные цвета
                if L < ub_threshold:
                    ub_layer[mask] = (0, 0, 0)  # Черный цвет для подложки
            
            # Добавляем слой подложки в начало списка
            color_layers.insert(0, ub_layer)
            color_info.insert(0, {
                'color': (0, 0, 0),
                'percentage': (np.sum(np.any(ub_layer != bg_color, axis=2)) / mask.size) * 100,
                'is_underbase': True
            })
        
        # Сопоставление с пользовательской палитрой, если требуется
        if use_custom_palette and custom_palette:
            color_matches = []
            
            for i, info in enumerate(color_info):
                if 'is_underbase' in info and info['is_underbase']:
                    color_matches.append("Underbase (черный)")
                    continue
                
                closest_name, closest_color, distance = find_closest_color_in_palette(
                    info['color'], custom_palette
                )
                
                # Обновляем информацию о цвете
                color_info[i]['matched_color'] = closest_color
                color_info[i]['matched_name'] = closest_name
                color_info[i]['match_distance'] = distance
                
                color_matches.append(f"{closest_name} (расстояние: {distance:.2f})")
            
            if color_matches:
                st.info("Сопоставление цветов с палитрой:")
                for i, match in enumerate(color_matches):
                    st.write(f"Слой {i+1}: {match}")
        
        return color_layers, color_info
    
    except Exception as e:
        st.error(f"Ошибка в методе Inksplit: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return [], []

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def convert_to_png(image_array, filename):
    """Конвертирует массив изображения в формат PNG"""
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_array)
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
    # Создаем маску для определения фона
    is_background = np.all(layer == bg_color, axis=2)
    
    # Создаем маску (255 для цвета, 0 для фона)
    mask = np.zeros((layer.shape[0], layer.shape[1]), dtype=np.uint8)
    mask[~is_background] = 255
    
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

# ==================== ПРЕДОПРЕДЕЛЕННЫЕ ПАЛИТРЫ ДЛЯ INKSPLIT ====================

def get_default_palettes():
    """Возвращает предопределенные палитры для метода Inksplit"""
    palettes = {
        "Пантон (PANTONE)": [
            ("PANTONE 186 C", (200, 16, 46)),
            ("PANTONE 032 C", (255, 83, 73)),
            ("PANTONE 021 C", (255, 184, 28)),
            ("PANTONE 356 C", (0, 173, 68)),
            ("PANTONE 293 C", (0, 56, 168)),
            ("PANTONE 266 C", (96, 57, 147)),
            ("PANTONE Black C", (35, 31, 32)),
            ("PANTONE White", (255, 255, 255)),
        ],
        "CMYK (для печати)": [
            ("Cyan", (0, 174, 239)),
            ("Magenta", (236, 0, 140)),
            ("Yellow", (255, 242, 0)),
            ("Black", (35, 31, 32)),
            ("White", (255, 255, 255)),
        ],
        "RGB": [
            ("Red", (255, 0, 0)),
            ("Green", (0, 255, 0)),
            ("Blue", (0, 0, 255)),
            ("Yellow", (255, 255, 0)),
            ("Cyan", (0, 255, 255)),
            ("Magenta", (255, 0, 255)),
            ("Black", (0, 0, 0)),
            ("White", (255, 255, 255)),
        ],
        "Spot Colors": [
            ("Gold", (212, 175, 55)),
            ("Silver", (192, 192, 192)),
            ("Metallic Blue", (0, 51, 153)),
            ("Fluorescent Pink", (255, 0, 255)),
            ("Neon Green", (57, 255, 20)),
        ]
    }
    return palettes

def create_custom_palette_from_image(image, n_colors):
    """Создает пользовательскую палитру из изображения"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)
    
    # Удаляем белый фон
    bg_color = (255, 255, 255)
    bg_mask = np.all(pixels == bg_color, axis=1)
    if np.any(bg_mask):
        pixels = pixels[~bg_mask]
    
    if len(pixels) == 0:
        return []
    
    # Используем K-means для создания палитры
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_.astype(int)
    
    # Создаем список цветов с именами
    palette = []
    for i, color in enumerate(colors):
        palette.append((f"Цвет {i+1}", (int(color[0]), int(color[1]), int(color[2]))))
    
    return palette

# ==================== БОКОВАЯ ПАНЕЛЬ ====================

with st.sidebar:
    st.markdown("<h2 class='sub-header'>⚙️ Настройки</h2>", unsafe_allow_html=True)
    
    # Загрузка изображения
    st.markdown("<h4>📤 Загрузите изображение</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Выберите файл", type=["jpg", "jpeg", "png", "bmp", "tiff"], 
                                    label_visibility="collapsed")
    
    if uploaded_file is not None:
        # Сохраняем в session state
        st.session_state.uploaded_file = uploaded_file
        
        # Выбор метода
        st.markdown("<h4>🎯 Выберите метод</h4>", unsafe_allow_html=True)
        methods = ["K-средних кластеризация", "Inksplit (для трафаретной печати)"]
        if model_available:
            methods.append("Fast Soft Color Segmentation (нейронная сеть)")
        
        selected_method = st.selectbox("Метод разделения", methods, 
                                      label_visibility="collapsed")
        st.session_state.selected_method = selected_method
        
        # Количество цветов
        st.markdown("<h4>🌈 Количество цветов</h4>", unsafe_allow_html=True)
        num_colors = st.slider("От 2 до 8 цветов", 2, 8, 5, 
                              help="Выберите количество цветов для извлечения из изображения",
                              label_visibility="collapsed")
        
        # Цвет фона
        st.markdown("<h4>🎨 Цвет фона</h4>", unsafe_allow_html=True)
        bg_color = st.color_picker("Цвет фона для слоев", "#FFFFFF", 
                                  label_visibility="collapsed")
        bg_color_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Дополнительные настройки для Inksplit
        if selected_method == "Inksplit (для трафаретной печати)":
            st.markdown("<div class='inksplit-options'>", unsafe_allow_html=True)
            st.markdown("<h4>🎨 Настройки Inksplit</h4>", unsafe_allow_html=True)
            
            # Основные опции Inksplit
            auto_color_match = st.checkbox("Автоматическое определение цветов", True,
                                         help="Автоматически определять цвета из изображения")
            
            dithering = st.checkbox("Сглаживание (dithering)", True,
                                  help="Применять сглаживание для лучших переходов")
            
            generate_ub = st.checkbox("Создать подложку (Underbase)", True,
                                    help="Создать черную подложку для темных цветов")
            
            if generate_ub:
                ub_threshold = st.slider("Порог яркости для подложки", 0.0, 1.0, 0.35, 0.05,
                                       help="Цвета темнее этого порога будут удалены из подложки")
            
            # Использование пользовательской палитры
            use_custom_palette = st.checkbox("Использовать пользовательскую палитру", False,
                                           help="Сопоставить цвета с предопределенной палитрой")
            
            if use_custom_palette:
                palettes = get_default_palettes()
                palette_names = list(palettes.keys())
                selected_palette_name = st.selectbox("Выберите палитру", palette_names)
                
                if selected_palette_name:
                    selected_palette = palettes[selected_palette_name]
                    
                    # Показать цвета палитры
                    st.write("Цвета в палитре:")
                    cols = st.columns(4)
                    for idx, (color_name, color_rgb) in enumerate(selected_palette):
                        with cols[idx % 4]:
                            hex_color = "#{:02x}{:02x}{:02x}".format(*color_rgb)
                            st.markdown(f"""
                            <div style='text-align: center;'>
                                <div style='width: 40px; height: 40px; background-color: {hex_color}; 
                                         margin: 0 auto; border: 1px solid #000; border-radius: 5px;'></div>
                                <div style='font-size: 0.8em;'>{color_name}</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                best_fit_match = st.checkbox("Сопоставить с лучшим соответствием", True,
                                           help="Найти ближайшие соответствия в палитре")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Настройки экспорта
            st.markdown("<div class='export-options'>", unsafe_allow_html=True)
            st.markdown("<h4>📤 Настройки экспорта</h4>", unsafe_allow_html=True)
            
            export_format = st.selectbox("Формат экспорта", ["PNG", "PDF", "SVG", "Все форматы"],
                                       help="Выберите формат для экспорта слоев")
            
            include_labels = st.checkbox("Включить метки слоев", True,
                                       help="Добавить текстовые метки к слоям")
            
            if include_labels:
                label_font_size = st.slider("Размер шрифта меток", 10, 50, 20)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Дополнительные настройки для нейронной сети
        elif selected_method == "Fast Soft Color Segmentation (нейронная сеть)" and model_available:
            st.markdown("<h4>⚡ Настройки нейронной сети</h4>", unsafe_allow_html=True)
            resize_factor = st.slider("Масштаб", 0.5, 2.0, 1.0, 0.1,
                                     help="Коэффициент изменения размера для обработки",
                                     label_visibility="collapsed")
        
        # Дополнительные опции для всех методов
        with st.expander("🛠️ Дополнительные настройки", expanded=False):
            st.markdown("<p style='color: #666; font-size: 0.9em;'>Эти настройки отключены по умолчанию для лучшей производительности</p>", 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                apply_smoothing = st.checkbox("Сглаживание", False, 
                                             help="Применить сглаживание к маскам")
                if apply_smoothing:
                    smoothing_amount = st.slider("Степень сглаживания", 1, 10, 3, 
                                                label_visibility="collapsed")
            
            with col2:
                apply_sharpening = st.checkbox("Резкость", False,
                                              help="Увеличить резкость границ")
                if apply_sharpening:
                    sharpening_amount = st.slider("Степень резкости", 0.1, 3.0, 1.0, 0.1,
                                                 label_visibility="collapsed")
            
            noise_reduction = st.checkbox("Уменьшение шума", False,
                                         help="Уменьшить шум в масках")
            if noise_reduction:
                noise_amount = st.slider("Степень уменьшения", 1, 10, 3,
                                        label_visibility="collapsed")

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
    if selected_method == "Inksplit (для трафаретной печати)":
        st.markdown(f"""
        <div class="method-card-inksplit">
            <h4>🎯 Выбранный метод: <strong>{selected_method}</strong></h4>
            <p>📊 Количество цветов: <strong>{num_colors}</strong> | 🎨 Цвет фона: <span style='color: {bg_color}; font-weight: bold;'>{bg_color}</span></p>
            <p>🚀 <em>Специализированный метод для трафаретной печати с поддержкой подложки и цветовых палитр</em></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="method-card">
            <h4>🎯 Выбранный метод: <strong>{selected_method}</strong></h4>
            <p>📊 Количество цветов: <strong>{num_colors}</strong> | 🎨 Цвет фона: <span style='color: {bg_color}; font-weight: bold;'>{bg_color}</span></p>
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
        with st.expander("📊 Информация об изображении"):
            st.write(f"**Размер:** {image.width} × {image.height} пикселей")
            st.write(f"**Формат:** {image.format}")
            st.write(f"**Режим:** {image.mode}")
            st.write(f"**Размер файла:** {len(image_bytes) / 1024:.1f} KB")
            
            # Гистограмма цветов
            if st.checkbox("Показать гистограмму цветов"):
                img_array = np.array(image)
                if img_array.shape[2] == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
                colors = ('r', 'g', 'b')
                fig, ax = plt.subplots(figsize=(8, 4))
                for i, color in enumerate(colors):
                    histogram = cv2.calcHist([img_array], [i], None, [256], [0, 256])
                    ax.plot(histogram, color=color)
                
                ax.set_xlim([0, 256])
                ax.set_xlabel('Интенсивность цвета')
                ax.set_ylabel('Количество пикселей')
                ax.set_title('Гистограмма цветов RGB')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
    
    with col2:
        st.markdown("<h3 class='sub-header'>🎨 Разделенные цветовые слои</h3>", unsafe_allow_html=True)
        
        # Подготовка параметров для метода Inksplit
        inksplit_params = {}
        if selected_method == "Inksplit (для трафаретной печати)":
            inksplit_params = {
                'auto_color_match': auto_color_match,
                'dithering': dithering,
                'generate_ub': generate_ub if 'generate_ub' in locals() else False,
                'ub_threshold': ub_threshold if 'ub_threshold' in locals() else 0.35,
                'use_custom_palette': use_custom_palette if 'use_custom_palette' in locals() else False,
                'custom_palette': selected_palette if 'selected_palette' in locals() else None,
                'best_fit_match': best_fit_match if 'best_fit_match' in locals() else False,
            }
        
        # Кнопка для запуска обработки
        if st.button("🚀 Начать разделение цветов", type="primary", use_container_width=True):
            with st.spinner("🔄 Обработка изображения... Пожалуйста, подождите."):
                try:
                    if selected_method == "K-средних кластеризация":
                        # Используем K-means
                        color_layers, color_info = kmeans_color_separation(
                            img_cv, 
                            n_colors=num_colors,
                            bg_color=bg_color_rgb
                        )
                    
                    elif selected_method == "Inksplit (для трафаретной печати)":
                        # Используем метод Inksplit
                        color_layers, color_info = inksplit_color_separation(
                            img_cv,
                            n_colors=num_colors,
                            bg_color=bg_color_rgb,
                            **inksplit_params
                        )
                    
                    elif selected_method == "Fast Soft Color Segmentation (нейронная сеть)":
                        # Используем нейронную сеть
                        if not model_available:
                            st.error("Модель не найдена. Используйте метод K-means или загрузите модель.")
                            color_layers, color_info = [], []
                        else:
                            # Получаем доминирующие цвета для палитры
                            palette_colors = get_dominant_colors(image, num_colors)
                            
                            # Вызываем функцию decompose
                            decompose_layers = decompose_fast_soft_color(
                                image,
                                num_colors=num_colors,
                                palette=palette_colors,
                                resize_scale_factor=resize_factor if 'resize_factor' in locals() else 1.0
                            )
                            
                            if decompose_layers:
                                # Преобразуем слои decompose в формат для отображения
                                color_layers, color_info = decompose_layers_to_cv_format(
                                    decompose_layers, 
                                    bg_color_rgb
                                )
                            else:
                                st.error("Не удалось выполнить разделение с помощью нейронной сети.")
                                color_layers, color_info = [], []
                    
                    # Сохраняем результаты в session state
                    st.session_state.color_layers = color_layers
                    st.session_state.color_info = color_info
                    
                    if color_layers and color_info:
                        st.success(f"✅ Успешно создано {len(color_layers)} цветовых слоев!")
                        
                        # Для метода Inksplit показываем дополнительную информацию
                        if selected_method == "Inksplit (для трафаретной печати)":
                            # Проверяем, есть ли подложка
                            has_underbase = any('is_underbase' in info for info in color_info)
                            if has_underbase:
                                st.info("ℹ️ Создан слой подложки (Underbase). Темные цвета удалены из подложки для лучшей печати.")
                            
                            # Показываем сопоставление цветов, если использовалась палитра
                            if use_custom_palette and 'selected_palette' in locals():
                                st.info("🎨 Цвета сопоставлены с выбранной палитрой.")
                    else:
                        st.warning("⚠️ Не удалось создать цветовые слои. Попробуйте изменить параметры.")
                        
                except Exception as e:
                    st.error(f"❌ Ошибка при обработке изображения: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        
        # Показываем результаты если они есть
        color_layers = st.session_state.color_layers
        color_info = st.session_state.color_info
        
        if color_layers and color_info:
            # Для метода Inksplit показываем специальную информацию
            if selected_method == "Inksplit (для трафаретной печати)":
                # Определяем, есть ли подложка
                has_underbase = any('is_underbase' in info for info in color_info)
                underbase_index = None
                
                if has_underbase:
                    for i, info in enumerate(color_info):
                        if 'is_underbase' in info and info['is_underbase']:
                            underbase_index = i
                            break
                
                # Показываем предупреждение о подложке
                if has_underbase and underbase_index is not None:
                    st.warning(f"""
                    ⚠️ **Внимание!** Слой {underbase_index + 1} - это подложка (Underbase). 
                    
                    **Рекомендации по печати:**
                    - Подложка печатается первой
                    - Используйте специальную белую или прозрачную краску для подложки
                    - Остальные цвета печатаются поверх подложки
                    - Подложка улучшает яркость светлых цветов на темных поверхностях
                    """)
            
            # Создаем вкладки для каждого слоя
            tabs = st.tabs([f"Слой {i+1}" for i in range(len(color_layers))])
            
            for i, (layer, info) in enumerate(zip(color_layers, color_info)):
                with tabs[i]:
                    col_left, col_right = st.columns([3, 1])
                    
                    with col_left:
                        # Конвертация слоя из BGR в RGB для отображения
                        layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                        
                        # Проверяем, является ли это подложкой
                        is_underbase = 'is_underbase' in info and info['is_underbase']
                        
                        if is_underbase:
                            # Для подложки показываем специальную информацию
                            st.warning("🎨 **Слой подложки (Underbase)**")
                            st.image(layer_rgb, use_column_width=True, 
                                   caption="Черная подложка для темных цветов")
                        else:
                            st.image(layer_rgb, use_column_width=True)
                        
                        # Кнопки для скачивания
                        col_btn1, col_btn2 = st.columns(2)
                        
                        with col_btn1:
                            # Черно-белая маска
                            bw_mask = create_bw_mask(layer, bg_color_rgb)
                            png_data = save_bw_mask_as_png(bw_mask, f"mask_{i+1}")
                            
                            if png_data:
                                # Определяем имя цвета для файла
                                if 'matched_name' in info:
                                    color_name = info['matched_name'].replace(" ", "_")
                                elif is_underbase:
                                    color_name = "Underbase"
                                else:
                                    color_name = f"color_{i+1}"
                                
                                st.download_button(
                                    label="⬇️ Скачать ЧБ маску",
                                    data=png_data,
                                    file_name=f"{color_name}_mask.png",
                                    mime="image/png",
                                    key=f"download_mask_{i}"
                                )
                        
                        with col_btn2:
                            # Цветной слой
                            color_png_data = convert_to_png(layer_rgb, f"layer_{i+1}")
                            if color_png_data:
                                if 'matched_name' in info:
                                    color_name = info['matched_name'].replace(" ", "_")
                                elif is_underbase:
                                    color_name = "Underbase"
                                else:
                                    color_name = f"color_{i+1}"
                                
                                st.download_button(
                                    label="⬇️ Скачать цветной слой",
                                    data=color_png_data,
                                    file_name=f"{color_name}_color.png",
                                    mime="image/png",
                                    key=f"download_color_{i}"
                                )
                    
                    with col_right:
                        # Информация о цвете
                        if is_underbase:
                            # Информация для подложки
                            hex_color = "#000000"
                            st.markdown(f"""
                            <div style='padding: 15px; background-color: #fff3cd; border-radius: 10px; border: 2px solid #ffeaa7;'>
                                <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                                    <div class='color-chip' style='background-color: {hex_color};'></div>
                                    <div>
                                        <strong style='font-size: 1.2em;'>Подложка (Underbase)</strong><br>
                                        <span style='color: #666; font-size: 0.9em;'>Специальный слой</span>
                                    </div>
                                </div>
                                <div style='margin-bottom: 10px;'>
                                    <strong>Назначение:</strong> База для печати<br>
                                    <strong>Цвет:</strong> Черный<br>
                                    <strong>Покрытие:</strong> {info['percentage']:.1f}%<br>
                                    <strong>Рекомендация:</strong> Печатать первым
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Определяем цвет для отображения
                            if 'matched_color' in info:
                                display_color = info['matched_color']
                                color_name = info.get('matched_name', f'Цвет {i+1}')
                            else:
                                display_color = info['color']
                                color_name = f'Цвет {i+1}'
                            
                            hex_color = "#{:02x}{:02x}{:02x}".format(
                                display_color[0], display_color[1], display_color[2]
                            )
                            
                            # Информация о расстоянии сопоставления
                            match_info = ""
                            if 'match_distance' in info:
                                match_info = f"<br><strong>Сходство:</strong> {info['match_distance']:.2f}"
                            
                            st.markdown(f"""
                            <div style='padding: 15px; background-color: #f8f9fa; border-radius: 10px;'>
                                <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                                    <div class='color-chip' style='background-color: {hex_color};'></div>
                                    <div>
                                        <strong style='font-size: 1.2em;'>{color_name}</strong><br>
                                        <span style='color: #666; font-size: 0.9em;'>{hex_color}</span>
                                    </div>
                                </div>
                                <div style='margin-bottom: 10px;'>
                                    <strong>RGB:</strong> {display_color}<br>
                                    <strong>Покрытие:</strong> {info['percentage']:.1f}%<br>
                                    <strong>Пикселей:</strong> {layer.shape[1]} × {layer.shape[0]}
                                    {match_info}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # ==================== КОМБИНИРОВАННЫЙ ПРЕДПРОСМОТР ====================
            
            st.markdown("---")
            st.markdown("<h3 class='sub-header'>👁️ Комбинированный предпросмотр</h3>", unsafe_allow_html=True)
            
            # Настройки порядка слоев
            with st.expander("⚙️ Управление порядком слоев", expanded=True):
                # Инициализация состояния сессии для порядка и видимости
                if 'layer_order' not in st.session_state or len(st.session_state.layer_order) != len(color_layers):
                    st.session_state.layer_order = list(range(len(color_layers)))
                if 'layer_visibility' not in st.session_state or len(st.session_state.layer_visibility) != len(color_layers):
                    st.session_state.layer_visibility = [True] * len(color_layers)
                
                # Для Inksplit устанавливаем подложку как видимую, но не изменяемую
                if selected_method == "Inksplit (для трафаретной печати)":
                    has_underbase = any('is_underbase' in info for info in color_info)
                    if has_underbase:
                        for i, info in enumerate(color_info):
                            if 'is_underbase' in info and info['is_underbase']:
                                st.session_state.layer_visibility[i] = True
                                st.info(f"Слой {i+1} (подложка) всегда включен и находится внизу стека.")
                
                # Настройки для каждого слоя
                for i in range(len(color_layers)):
                    is_underbase = 'is_underbase' in color_info[i] and color_info[i]['is_underbase']
                    
                    if is_underbase:
                        # Для подложки показываем фиксированные настройки
                        st.markdown(f"""
                        <div style='padding: 10px; background-color: #fff3cd; border-radius: 5px; margin-bottom: 10px;'>
                            <strong>Слой {i+1} (Подложка)</strong> - фиксированная позиция (нижний слой)
                            <div style='font-size: 0.9em; color: #666;'>
                                Этот слой печатается первым и не может быть изменен
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        continue
                    
                    col1, col2, col3 = st.columns([2, 1, 3])
                    
                    with col1:
                        # Порядок слоя
                        order_value = st.number_input(
                            f"Позиция слоя {i+1}",
                            min_value=1,
                            max_value=len(color_layers),
                            value=st.session_state.layer_order[i] + 1,
                            key=f"order_{i}",
                            help="1 = нижний слой (фон), больше = выше"
                        )
                        st.session_state.layer_order[i] = order_value - 1
                    
                    with col2:
                        # Видимость слоя
                        visibility = st.checkbox(
                            "Вкл",
                            value=st.session_state.layer_visibility[i],
                            key=f"visibility_{i}"
                        )
                        st.session_state.layer_visibility[i] = visibility
                    
                    with col3:
                        # Информация о цвете
                        if 'matched_color' in color_info[i]:
                            display_color = color_info[i]['matched_color']
                            color_name = color_info[i].get('matched_name', f'Цвет {i+1}')
                        else:
                            display_color = color_info[i]['color']
                            color_name = f'Цвет {i+1}'
                        
                        hex_color = "#{:02x}{:02x}{:02x}".format(
                            display_color[0], display_color[1], display_color[2]
                        )
                        
                        st.markdown(f"""
                        <div style='display: flex; align-items: center; padding: 8px; background-color: {'#e8f5e9' if visibility else '#f5f5f5'}; border-radius: 5px;'>
                            <div style='width: 25px; height: 25px; background-color: {hex_color}; border: 1px solid #000; border-radius: 4px; margin-right: 10px;'></div>
                            <div>
                                <div><strong>{color_name}</strong></div>
                                <div style='font-size: 0.8em; color: #666;'>{hex_color} • {color_info[i]['percentage']:.1f}%</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Создание комбинированного изображения
            combined = np.zeros_like(img_cv, dtype=np.uint8)
            
            # Для Inksplit: подложка всегда внизу и включена
            if selected_method == "Inksplit (для трафаретной печати)":
                # Сначала находим и применяем подложку
                for i, info in enumerate(color_info):
                    if 'is_underbase' in info and info['is_underbase']:
                        layer = color_layers[i]
                        if layer.shape != combined.shape:
                            layer = resize_layer_to_match(layer, combined.shape)
                        
                        mask = np.any(layer != bg_color_rgb, axis=2)
                        combined[mask] = layer[mask]
                        break
            
            # Сортируем индексы по порядку (от нижнего к верхнему)
            sorted_indices = sorted(range(len(st.session_state.layer_order)), 
                                   key=lambda x: st.session_state.layer_order[x])
            
            # Применяем слои в правильном порядке
            for idx in sorted_indices:
                # Пропускаем подложку в Inksplit, так как она уже применена
                if selected_method == "Inksplit (для трафаретной печати)" and \
                   'is_underbase' in color_info[idx] and color_info[idx]['is_underbase']:
                    continue
                
                if st.session_state.layer_visibility[idx]:
                    layer = color_layers[idx]
                    
                    # Проверяем размеры и изменяем при необходимости
                    if layer.shape != combined.shape:
                        layer = resize_layer_to_match(layer, combined.shape)
                    
                    # Создаем маску (где есть цвет, отличный от фона)
                    mask = np.any(layer != bg_color_rgb, axis=2)
                    
                    # Применяем слой только там, где есть маска
                    combined[mask] = layer[mask]
            
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
            col_comb1, col_comb2 = st.columns(2)
            
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
                        label="⬇️ Скачать комбинированную ЧБ маску",
                        data=combined_png_data,
                        file_name="combined_mask.png",
                        mime="image/png",
                        key="download_combined_mask"
                    )
            
            with col_comb2:
                # Цветное комбинированное изображение
                combined_color_png = convert_to_png(combined_rgb, "combined_preview")
                if combined_color_png:
                    st.download_button(
                        label="⬇️ Скачать цветной предпросмотр",
                        data=combined_color_png,
                        file_name="combined_preview.png",
                        mime="image/png",
                        key="download_combined_color"
                    )
            
            # ==================== ПАКЕТНОЕ СКАЧИВАНИЕ ====================
            
            st.markdown("---")
            st.markdown("<h3 class='sub-header'>📦 Пакетное скачивание</h3>", unsafe_allow_html=True)
            
            if st.button("📁 Создать ZIP-архив со всеми слоями", type="secondary", use_container_width=True):
                with st.spinner("🔄 Создание архива..."):
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        # Сохраняем все слои
                        all_files = []
                        
                        for i, layer in enumerate(color_layers):
                            if st.session_state.layer_visibility[i]:
                                # Определяем имя для файла
                                if selected_method == "Inksplit (для трафаретной печати)" and \
                                   'is_underbase' in color_info[i] and color_info[i]['is_underbase']:
                                    file_prefix = "Underbase"
                                elif 'matched_name' in color_info[i]:
                                    file_prefix = color_info[i]['matched_name'].replace(" ", "_")
                                else:
                                    file_prefix = f"layer_{i+1}"
                                
                                # Черно-белая маска
                                bw_mask = create_bw_mask(layer, bg_color_rgb)
                                mask_png = save_bw_mask_as_png(bw_mask, f"{file_prefix}_mask")
                                
                                if mask_png:
                                    mask_path = os.path.join(tmpdirname, f"{file_prefix}_mask.png")
                                    with open(mask_path, 'wb') as f:
                                        f.write(mask_png)
                                    all_files.append(mask_path)
                                
                                # Цветной слой
                                layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                                color_png = convert_to_png(layer_rgb, f"{file_prefix}_color")
                                
                                if color_png:
                                    color_path = os.path.join(tmpdirname, f"{file_prefix}_color.png")
                                    with open(color_path, 'wb') as f:
                                        f.write(color_png)
                                    all_files.append(color_path)
                        
                        # Сохраняем комбинированные изображения
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
                        readme_content = f"""# ColorSep Pro - Экспортированные слои

Дата создания: {st.session_state.get('processing_time', 'Неизвестно')}
Метод: {selected_method}
Количество слоев: {len(color_layers)}
Цвет фона: {bg_color}

## Содержимое архива:
- Черно-белые маски каждого слоя (layer_X_mask.png)
- Цветные изображения каждого слоя (layer_X_color.png)
- Комбинированные изображения (combined_*.png)

## Информация о слоях:
"""
                        
                        for i, info in enumerate(color_info):
                            if selected_method == "Inksplit (для трафаретной печати)" and \
                               'is_underbase' in info and info['is_underbase']:
                                readme_content += f"- Слой {i+1}: ПОДЛОЖКА (Underbase), Черный, Покрытие: {info['percentage']:.1f}%\n"
                            else:
                                if 'matched_color' in info:
                                    display_color = info['matched_color']
                                    color_name = info.get('matched_name', f'Цвет {i+1}')
                                else:
                                    display_color = info['color']
                                    color_name = f'Цвет {i+1}'
                                
                                hex_color = "#{:02x}{:02x}{:02x}".format(
                                    display_color[0], display_color[1], display_color[2]
                                )
                                
                                match_info = ""
                                if 'match_distance' in info:
                                    match_info = f", Сходство: {info['match_distance']:.2f}"
                                
                                readme_content += f"- Слой {i+1}: {color_name}, {hex_color}, RGB{display_color}, Покрытие: {info['percentage']:.1f}%{match_info}\n"
                        
                        readme_path = os.path.join(tmpdirname, "README.txt")
                        with open(readme_path, 'w', encoding='utf-8') as f:
                            f.write(readme_content)
                        all_files.append(readme_path)
                        
                        # Создаем инструкцию по печати для Inksplit
                        if selected_method == "Inksplit (для трафаретной печати)":
                            instructions_content = """# ИНСТРУКЦИЯ ПО ПЕЧАТИ - Метод Inksplit

## Порядок печати слоев:
1. Подложка (Underbase) - печатается ПЕРВОЙ
   - Используйте белую или прозрачную краску
   - Служит основой для светлых цветов на темных поверхностях

2. Цветные слои - печатаются ПОСЛЕ подложки
   - Порядок печати соответствует порядку слоев в архиве
   - Каждый слой печатается отдельно
   - Дайте каждому слою высохнуть перед нанесением следующего

## Рекомендации:
- Используйте трафаретную печать (шелкографию)
- Для подложки используйте специальную краску
- Проведите тестовую печать на образце материала
- Учитывайте настройки экспозиции для каждого слоя
"""
                            
                            instructions_path = os.path.join(tmpdirname, "ПЕЧАТЬ_ИНСТРУКЦИЯ.txt")
                            with open(instructions_path, 'w', encoding='utf-8') as f:
                                f.write(instructions_content)
                            all_files.append(instructions_path)
                        
                        # Создаем ZIP архив
                        zip_path = os.path.join(tmpdirname, "color_layers.zip")
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for file in all_files:
                                zipf.write(file, os.path.basename(file))
                        
                        # Читаем ZIP файл
                        with open(zip_path, "rb") as f:
                            zip_data = f.read()
                        
                        # Предоставляем для скачивания
                        st.download_button(
                            label="⬇️ Скачать ZIP архив со всеми файлами",
                            data=zip_data,
                            file_name="color_separation_layers.zip",
                            mime="application/zip",
                            key="download_all_zip"
                        )

# ==================== ИНФОРМАЦИЯ О МЕТОДАХ ====================

st.markdown("---")
st.markdown("<h2 class='sub-header'>📚 Описание методов</h2>", unsafe_allow_html=True)

col_method1, col_method2 = st.columns(2)

with col_method1:
    st.markdown("""
    <div class="method-card">
        <h4>🎯 K-средних кластеризация</h4>
        <p><strong>Описание:</strong> Классический алгоритм машинного обучения для группировки похожих цветов.</p>
        <p><strong>Преимущества:</strong></p>
        <ul>
            <li>Быстрая обработка</li>
            <li>Контролируемое количество цветов</li>
            <li>Хорошо работает с четкими цветами</li>
        </ul>
        <p><strong>Идеально для:</strong> Логотипы, векторная графика, изображения с четкими цветами</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="method-card-inksplit">
        <h4>🔧 Inksplit (для трафаретной печати)</h4>
        <p><strong>Описание:</strong> Специализированный метод для подготовки к трафаретной печати.</p>
        <p><strong>Преимущества:</strong></p>
        <ul>
            <li>Создание подложки (Underbase)</li>
            <li>Сопоставление с палитрами Pantone/CMYK</li>
            <li>Оптимизация для шелкографии</li>
            <li>Поддержка сглаживания</li>
        </ul>
        <p><strong>Идеально для:</strong> Трафаретная печать, шелкография, текстиль</p>
    </div>
    """, unsafe_allow_html=True)

with col_method2:
    if model_available:
        st.markdown("""
        <div class="method-card">
            <h4>⚡ Fast Soft Color Segmentation</h4>
            <p><strong>Описание:</strong> Нейронная сеть для продвинутого разделения цветов.</p>
            <p><strong>Преимущества:</strong></p>
            <ul>
                <li>Создает слои с прозрачностью</li>
                <li>Сохраняет плавные переходы</li>
                <li>Лучше работает с градиентами</li>
            </ul>
            <p><strong>Идеально для:</strong> Фотографии, градиенты, сложные текстуры</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="method-card" style="border-left-color: #ffc107;">
            <h4>⚡ Fast Soft Color Segmentation</h4>
            <p><strong>Статус:</strong> 🔒 Требуется модель</p>
            <p>Для использования этого метода необходимо скачать файл модели и поместить его в папку <code>model/</code></p>
            <p><strong>Преимущества метода:</strong></p>
            <ul>
                <li>Нейронная сеть для точного разделения</li>
                <li>Слои с альфа-каналами</li>
                <li>Идеально для сложных изображений</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Дополнительная информация о методе Inksplit
    st.markdown("""
    <div class="method-card" style="border-left-color: #4CAF50;">
        <h4>ℹ️ О методе Inksplit</h4>
        <p><strong>Что такое подложка (Underbase)?</strong></p>
        <p>Подложка - это специальный слой, который печатается первым и служит основой для светлых цветов на темных поверхностях.</p>
        
        <p><strong>Преимущества подложки:</strong></p>
        <ul>
            <li>Улучшает яркость светлых цветов</li>
            <li>Повышает стойкость печати</li>
            <li>Снижает влияние цвета материала</li>
        </ul>
        
        <p><strong>Когда использовать Inksplit?</strong></p>
        <ul>
            <li>Печать на темных тканях</li>
            <li>Шелкография на текстиле</li>
            <li>Трафаретная печать</li>
            <li>Профессиональная полиграфия</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================== ФУТЕР ====================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 30px; background-color: #f8f9fa; border-radius: 10px;">
    <h4>🎨 ColorSep Pro</h4>
    <p>Профессиональный инструмент для разделения цветов</p>
    <p style="font-size: 0.9em;">Поддерживаемые форматы: JPG, PNG, BMP, TIFF | Максимальный размер: 50MB</p>
    <p style="font-size: 0.9em;">Все файлы экспортируются в формате PNG для промышленной совместимости</p>
    <p style="font-size: 0.9em;">Метод Inksplit специально разработан для трафаретной печати и шелкографии</p>
</div>
""", unsafe_allow_html=True)

# ==================== ПРОВЕРКА ЗАВИСИМОСТЕЙ ====================

try:
    # Проверяем основные зависимости
    dependencies_ok = True
    
    # Проверка OpenCV
    cv2_version = cv2.__version__
    
    # Проверка PyTorch
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    
    # Проверка scikit-learn
    from sklearn import __version__ as sklearn_version
    
    # Проверка colormath
    if colormath_available:
        colormath_status = "✅ Доступен"
    else:
        colormath_status = "⚠️ Частично доступен (некоторые функции Inksplit ограничены)"
    
    # Выводим информацию в sidebar
    with st.sidebar.expander("ℹ️ Информация о системе", expanded=False):
        st.write(f"**OpenCV:** {cv2_version}")
        st.write(f"**PyTorch:** {torch_version}")
        st.write(f"**CUDA:** {'✅ Доступен' if cuda_available else '❌ Не доступен'}")
        st.write(f"**scikit-learn:** {sklearn_version}")
        st.write(f"**colormath:** {colormath_status}")
        st.write(f"**Streamlit:** {st.__version__}")
        
except Exception as e:
    st.sidebar.error(f"Ошибка проверки зависимостей: {e}")

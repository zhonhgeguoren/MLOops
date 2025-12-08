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
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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

if 'processed_image' not in st.session_state:
    st.session_state.processed_image = False

# ==================== КЛАССЫ ДЛЯ МЕТОДА DECOMPOSE ====================

class ColorSepDataset(torch.utils.data.Dataset):
    """Датасет для разделения цветов"""
    def __init__(self, img, num_primary_color, palette):
        self.img = img.convert("RGB")
        self.palette_list = palette.reshape(-1, num_primary_color * 3)
        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        # Преобразуем PIL Image в numpy array
        np_img = np.array(self.img)
        np_img = np_img.transpose((2, 0, 1))  # HWC to CHW
        target_img = np_img / 255.0  # Нормализуем к [0, 1]

        # Создаем слои первичных цветов
        primary_color_layers = self._create_primary_color_layers(
            self.palette_list[index], target_img
        )

        # Преобразуем в тензоры PyTorch
        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color_layers.astype(np.float32))

        return target_img, primary_color_layers

    def __len__(self):
        return 1

    def _create_primary_color_layers(self, palette_values, target_img):
        # Преобразуем палитру в формат (num_colors, 3) и нормализуем
        primary_color = palette_values.reshape(self.num_primary_color, 3) / 255.0
        
        # Создаем слои для каждого цвета
        primary_color_layers = np.tile(
            np.ones_like(target_img), 
            (self.num_primary_color, 1, 1, 1)
        ) * primary_color.reshape(self.num_primary_color, 3, 1, 1)
        
        return primary_color_layers

class MaskGeneratorModel(nn.Module):
    """Модель генератора масок для разделения цветов"""
    def __init__(self, num_primary_color):
        super(MaskGeneratorModel, self).__init__()
        in_dim = 3 + num_primary_color * 3
        out_dim = num_primary_color

        # Энкодер
        self.conv1 = nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_dim * 4, in_dim * 8, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Декодер
        self.deconv1 = nn.ConvTranspose2d(
            in_dim * 8, in_dim * 4, kernel_size=3, stride=2, padding=1, 
            bias=False, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_dim * 8, in_dim * 2, kernel_size=3, stride=2, padding=1,
            bias=False, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_dim * 4, in_dim * 2, kernel_size=3, stride=2, padding=1,
            bias=False, output_padding=1
        )
        
        # Выходной слой
        self.conv4 = nn.Conv2d(in_dim * 2 + 3, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(in_dim * 2)
        self.bn2 = nn.BatchNorm2d(in_dim * 4)
        self.bn3 = nn.BatchNorm2d(in_dim * 8)
        self.bnde1 = nn.BatchNorm2d(in_dim * 4)
        self.bnde2 = nn.BatchNorm2d(in_dim * 2)
        self.bnde3 = nn.BatchNorm2d(in_dim * 2)
        self.bn4 = nn.BatchNorm2d(in_dim)

    def forward(self, target_img, primary_color_pack):
        # Конкатенируем входное изображение с палитрой цветов
        x = torch.cat((target_img, primary_color_pack), dim=1)

        # Прямой проход через энкодер
        h1 = F.relu(self.bn1(self.conv1(x)))  # /2
        h2 = F.relu(self.bn2(self.conv2(h1)))  # /4
        h3 = F.relu(self.bn3(self.conv3(h2)))  # /8
        
        # Прямой проход через декодер
        h4 = F.relu(self.bnde1(self.deconv1(h3)))  # *2
        h4 = torch.cat((h4, h2), dim=1)  # Skip connection
        h5 = F.relu(self.bnde2(self.deconv2(h4)))  # *2
        h5 = torch.cat((h5, h1), dim=1)  # Skip connection
        h6 = F.relu(self.bnde3(self.deconv3(h5)))  # *2
        h6 = torch.cat((h6, target_img), dim=1)  # Добавляем исходное изображение
        h7 = F.relu(self.bn4(self.conv4(h6)))
        
        # Выход - маски для каждого цвета
        return torch.sigmoid(self.conv5(h7))

# ==================== ФУНКЦИИ ДЛЯ МЕТОДА DECOMPOSE ====================

def extract_dominant_colors(image_pil, num_colors):
    """
    Извлекает доминирующие цвета из изображения с использованием K-means
    Возвращает цвета в правильном формате для нейронной сети
    """
    # Преобразуем PIL в numpy array
    img_array = np.array(image_pil)
    
    # Если изображение имеет альфа-канал, удаляем его
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Преобразуем в одномерный массив пикселей
    pixels = img_array.reshape(-1, 3)
    
    # Применяем K-means
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Получаем центры кластеров
    colors = kmeans.cluster_centers_.astype(np.float32)
    
    # Сортируем цвета по яркости для лучшего визуального представления
    # Конвертируем в YUV для вычисления яркости
    colors_yuv = cv2.cvtColor(colors.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2YUV)
    brightness = colors_yuv[0, :, 0]
    sorted_indices = np.argsort(brightness)[::-1]  # От самого яркого к темному
    colors = colors[sorted_indices]
    
    return colors

def decompose_image_neural(image_pil, num_colors, palette_colors=None, resize_factor=1.0):
    """
    Основная функция для разделения изображения с использованием нейронной сети
    Возвращает маски и цвета в правильном формате
    """
    if not model_available:
        st.error("Модель не найдена!")
        return [], []
    
    if num_colors < 2 or num_colors > 8:
        st.error(f"Количество цветов должно быть от 2 до 8. Получено: {num_colors}")
        return [], []
    
    try:
        # Если палитра не предоставлена, извлекаем доминирующие цвета
        if palette_colors is None:
            palette_colors = extract_dominant_colors(image_pil, num_colors)
        else:
            # Убедимся, что палитра в правильном формате
            palette_colors = np.array(palette_colors, dtype=np.float32)
            if len(palette_colors) != num_colors:
                st.warning(f"Предоставлено {len(palette_colors)} цветов, но требуется {num_colors}. Будут использованы доминирующие цвета.")
                palette_colors = extract_dominant_colors(image_pil, num_colors)
        
        # Подготавливаем палитру для модели
        palette_tensor = palette_colors.reshape(1, -1)  # (1, num_colors * 3)
        
        # Создаем датасет
        dataset = ColorSepDataset(image_pil, num_colors, palette_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        
        # Инициализируем модель
        device = torch.device("cpu")
        model = MaskGeneratorModel(num_colors).to(device)
        
        # Загружаем веса модели
        model_path = Path("model/mask_generator7.pth")
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()
        
        # Функция для обрезки изображения до размера, кратного 8
        def adjust_to_multiple_of_8(tensor, scale_factor=1.0):
            if scale_factor != 1.0:
                tensor = F.interpolate(
                    tensor, 
                    scale_factor=scale_factor, 
                    mode='bilinear', 
                    align_corners=False
                )
            
            h = tensor.size(2)
            w = tensor.size(3)
            h = h - (h % 8)
            w = w - (w % 8)
            return tensor[:, :, :h, :w]
        
        # Функция для нормализации масок
        def normalize_masks(masks):
            # Добавляем небольшую константу для избежания деления на ноль
            epsilon = 1e-8
            return masks / (masks.sum(dim=1, keepdim=True) + epsilon)
        
        with torch.no_grad():
            for target_img, primary_color_layers in dataloader:
                # Перемещаем данные на устройство
                target_img = target_img.to(device)
                primary_color_layers = primary_color_layers.to(device)
                
                # Подготавливаем входные данные
                target_img_adj = adjust_to_multiple_of_8(target_img, resize_factor)
                primary_color_pack = primary_color_layers.view(
                    1, -1, primary_color_layers.size(3), primary_color_layers.size(4)
                )
                primary_color_pack_adj = adjust_to_multiple_of_8(primary_color_pack, resize_factor)
                
                # Прямой проход через модель
                predicted_masks = model(target_img_adj, primary_color_pack_adj)
                
                # Нормализуем маски
                normalized_masks = normalize_masks(predicted_masks)
                
                # Восстанавливаем исходный размер
                if resize_factor != 1.0:
                    normalized_masks = F.interpolate(
                        normalized_masks,
                        size=(image_pil.height, image_pil.width),
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Преобразуем в numpy
                masks_np = normalized_masks[0].cpu().numpy()  # (num_colors, H, W)
                
                # Преобразуем цвета палитры в uint8
                colors_uint8 = palette_colors.astype(np.uint8)
                
                return masks_np, colors_uint8
        
        return [], []
        
    except Exception as e:
        st.error(f"Ошибка при выполнении нейронного разделения: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return [], []

def create_color_layers_from_masks(original_image_cv, masks, colors, bg_color=(255, 255, 255)):
    """
    Создает цветные слои из масок и цветов
    Возвращает слои и информацию о цветах
    """
    color_layers = []
    color_info = []
    
    # Преобразуем оригинальное изображение в RGB для расчетов
    original_rgb = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)
    h, w = original_rgb.shape[:2]
    
    for i, (mask, color) in enumerate(zip(masks, colors)):
        # Нормализуем маску к диапазону [0, 1]
        mask_normalized = mask / np.max(mask) if np.max(mask) > 0 else mask
        
        # Создаем 3-канальную маску
        mask_3d = np.stack([mask_normalized] * 3, axis=2)
        
        # Создаем слой с указанным цветом
        color_layer = np.zeros((h, w, 3), dtype=np.uint8)
        color_layer[:, :] = color  # Заполняем весь слой цветом
        
        # Применяем маску: где маска = 1, там цвет слоя, где 0 - цвет фона
        # Но для плавных переходов используем взвешенное смешение
        bg_layer = np.full((h, w, 3), bg_color, dtype=np.uint8)
        
        # Взвешенное смешение
        for c in range(3):
            color_layer[:, :, c] = (
                color_layer[:, :, c] * mask_3d[:, :, c] + 
                bg_layer[:, :, c] * (1 - mask_3d[:, :, c])
            ).astype(np.uint8)
        
        # Конвертируем обратно в BGR для OpenCV
        color_layer_bgr = cv2.cvtColor(color_layer, cv2.COLOR_RGB2BGR)
        
        # Вычисляем информацию о цвете
        # Используем медианный цвет там, где маска достаточно сильная
        mask_threshold = mask_normalized > 0.1
        if np.any(mask_threshold):
            # Получаем цвета из оригинального изображения там, где есть маска
            masked_colors = original_rgb[mask_threshold]
            
            # Вычисляем медианный цвет
            median_color = np.median(masked_colors, axis=0).astype(int)
            # Преобразуем RGB в BGR
            median_color_bgr = (median_color[2], median_color[1], median_color[0])
            
            # Процент покрытия
            coverage = np.sum(mask_threshold) / mask_threshold.size * 100
            
            # Интенсивность цвета в этом слое
            color_intensity = np.mean(mask_normalized[mask_threshold]) * 100 if np.any(mask_threshold) else 0
        else:
            median_color_bgr = bg_color
            coverage = 0
            color_intensity = 0
        
        # Сохраняем слой и информацию
        color_layers.append(color_layer_bgr)
        color_info.append({
            'color': median_color_bgr,
            'coverage': coverage,
            'intensity': color_intensity,
            'target_color': tuple(color[::-1]),  # BGR to RGB
            'mask': mask_normalized
        })
    
    return color_layers, color_info

# ==================== ФУНКЦИИ ДЛЯ МЕТОДА K-MEANS ====================

def kmeans_color_separation(img_cv, n_colors=5, bg_color=(255, 255, 255)):
    """
    Разделение цветов с использованием алгоритма K-means
    """
    if n_colors < 2 or n_colors > 8:
        st.error(f"Количество цветов должно быть от 2 до 8. Получено: {n_colors}")
        return [], []
    
    try:
        # Преобразуем BGR в RGB для K-means
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Преобразуем в одномерный массив пикселей
        pixels = img_rgb.reshape(-1, 3)
        
        # Применяем K-means
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Получаем цвета кластеров
        cluster_colors = kmeans.cluster_centers_.astype(int)
        
        # Создаем слои
        color_layers = []
        color_info = []
        
        for i in range(n_colors):
            # Создаем бинарную маску для текущего кластера
            mask_flat = (labels == i)
            mask = mask_flat.reshape(h, w)
            
            # Создаем слой
            layer = np.full((h, w, 3), bg_color, dtype=np.uint8)
            layer[mask] = cluster_colors[i]
            
            # Конвертируем в BGR
            layer_bgr = cv2.cvtColor(layer, cv2.COLOR_RGB2BGR)
            
            # Вычисляем информацию о цвете
            coverage = np.sum(mask) / mask.size * 100
            
            # Если есть пиксели в кластере, вычисляем доминирующий цвет
            if np.any(mask):
                masked_pixels = img_rgb[mask]
                # Используем наиболее часто встречающийся цвет
                unique_colors, counts = np.unique(masked_pixels, axis=0, return_counts=True)
                dominant_color_idx = np.argmax(counts)
                dominant_color_rgb = unique_colors[dominant_color_idx]
                dominant_color_bgr = (dominant_color_rgb[2], dominant_color_rgb[1], dominant_color_rgb[0])
            else:
                dominant_color_bgr = bg_color
            
            color_layers.append(layer_bgr)
            color_info.append({
                'color': dominant_color_bgr,
                'coverage': coverage,
                'intensity': 100 if coverage > 0 else 0,
                'target_color': tuple(cluster_colors[i]),
                'mask': mask.astype(float)
            })
        
        return color_layers, color_info
    
    except Exception as e:
        st.error(f"Ошибка в методе K-means: {str(e)}")
        return [], []

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def create_bw_mask(mask_array, threshold=0.1):
    """
    Создает черно-белую маску из массива маски
    """
    # Применяем порог
    bw_mask = (mask_array > threshold).astype(np.uint8) * 255
    return bw_mask

def save_image_as_png(image_array, filename="image.png", dpi=300):
    """
    Сохраняет массив изображения как PNG
    """
    try:
        # Если изображение одноцветное (маска), используем grayscale
        if len(image_array.shape) == 2:
            plt.figure(figsize=(10, 10), dpi=dpi)
            plt.imshow(image_array, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            # Сохраняем в буфер
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close()
            buf.seek(0)
            return buf.getvalue()
        else:
            # Цветное изображение
            # Конвертируем BGR в RGB если нужно
            if image_array.shape[2] == 3:
                # Проверяем, является ли это BGR
                if image_array[0, 0, 0] > image_array[0, 0, 2]:  # Если синий > красного, вероятно BGR
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 10), dpi=dpi)
            plt.imshow(image_array)
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close()
            buf.seek(0)
            return buf.getvalue()
    except Exception as e:
        st.error(f"Ошибка при сохранении PNG: {e}")
        return None

def resize_image_to_match(image, target_shape):
    """
    Изменяет размер изображения до целевого размера
    """
    if image.shape[:2] == target_shape[:2]:
        return image
    
    return cv2.resize(image, (target_shape[1], target_shape[0]), 
                     interpolation=cv2.INTER_LINEAR)

def calculate_color_similarity(color1, color2):
    """
    Вычисляет сходство между двумя цветами (0-100%)
    """
    # Евклидово расстояние в цветовом пространстве
    diff = np.array(color1) - np.array(color2)
    distance = np.sqrt(np.sum(diff**2))
    # Нормализуем к 0-100% (максимальное расстояние ~441)
    similarity = max(0, 100 - (distance / 441 * 100))
    return similarity

# ==================== БОКОВАЯ ПАНЕЛЬ ====================

with st.sidebar:
    st.markdown("<h2 class='sub-header'>⚙️ Настройки</h2>", unsafe_allow_html=True)
    
    # Загрузка изображения
    st.markdown("<h4>📤 Загрузите изображение</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Выберите файл", type=["jpg", "jpeg", "png", "bmp", "tiff"], 
                                    label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
        # Выбор метода
        st.markdown("<h4>🎯 Выберите метод</h4>", unsafe_allow_html=True)
        methods = ["K-средних кластеризация"]
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
        bg_color_bgr = bg_color_rgb[::-1]  # RGB to BGR
        
        # Дополнительные настройки для нейронной сети
        if selected_method == "Fast Soft Color Segmentation (нейронная сеть)" and model_available:
            st.markdown("<h4>⚡ Настройки нейронной сети</h4>", unsafe_allow_html=True)
            resize_factor = st.slider("Масштаб обработки", 0.5, 2.0, 1.0, 0.1,
                                     help="Коэффициент изменения размера для обработки",
                                     label_visibility="collapsed")
            
            # Настройки маски
            with st.expander("🎭 Настройки масок", expanded=False):
                mask_threshold = st.slider("Порог маски", 0.0, 1.0, 0.1, 0.05,
                                          help="Минимальное значение маски для учета пикселя")
                mask_smoothing = st.checkbox("Сглаживание масок", True,
                                            help="Применить сглаживание к маскам")
                if mask_smoothing:
                    smoothing_kernel = st.slider("Размер ядра сглаживания", 1, 11, 3, 2,
                                                help="Размер ядра Гауссова фильтра")
        
        # Дополнительные опции
        with st.expander("🛠️ Дополнительные настройки", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                enhance_edges = st.checkbox("Усиление границ", False,
                                          help="Усилить границы между цветами")
                
                preserve_colors = st.checkbox("Сохранить оригинальные цвета", True,
                                            help="Использовать оригинальные цвета изображения")
            
            with col2:
                remove_noise = st.checkbox("Удаление шума", True,
                                          help="Удалить мелкие шумовые элементы")
                
                merge_similar = st.checkbox("Объединять похожие цвета", True,
                                          help="Автоматически объединять похожие цветовые слои")

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
    st.markdown(f"""
    <div class="method-card">
        <h4>🎯 Выбранный метод: <strong>{selected_method}</strong></h4>
        <p>📊 Количество цветов: <strong>{num_colors}</strong> | 🎨 Цвет фона: <span style='color: {bg_color}; font-weight: bold;'>{bg_color}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    # Чтение изображения
    image_bytes = uploaded_file.getvalue()
    image_pil = Image.open(io.BytesIO(image_bytes))
    
    # Конвертация в OpenCV формат
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    st.session_state.original_image_cv = img_cv
    
    with col1:
        st.markdown("<h3 class='sub-header'>📷 Исходное изображение</h3>", unsafe_allow_html=True)
        st.image(image_pil, use_column_width=True)
        
        # Информация об изображении
        with st.expander("📊 Информация об изображении"):
            st.write(f"**Размер:** {image_pil.width} × {image_pil.height} пикселей")
            st.write(f"**Формат:** {image_pil.format}")
            st.write(f"**Режим:** {image_pil.mode}")
            st.write(f"**Размер файла:** {len(image_bytes) / 1024:.1f} KB")
            
            # Гистограмма цветов
            if st.checkbox("Показать гистограмму цветов"):
                img_array = np.array(image_pil)
                if img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]
                
                fig, axes = plt.subplots(1, 3, figsize=(12, 3))
                colors = ['Red', 'Green', 'Blue']
                for i, (ax, color) in enumerate(zip(axes, colors)):
                    ax.hist(img_array[:, :, i].ravel(), bins=256, color=color.lower(), alpha=0.7)
                    ax.set_title(f'{color} Channel')
                    ax.set_xlim([0, 256])
                plt.tight_layout()
                st.pyplot(fig)
    
    with col2:
        st.markdown("<h3 class='sub-header'>🎨 Разделенные цветовые слои</h3>", unsafe_allow_html=True)
        
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
                    
                    elif selected_method == "Fast Soft Color Segmentation (нейронная сеть)":
                        # Используем нейронную сеть
                        if not model_available:
                            st.error("Модель не найдена. Используйте метод K-means или загрузите модель.")
                            color_layers, color_info = [], []
                        else:
                            # Выполняем нейронное разделение
                            masks, colors_rgb = decompose_image_neural(
                                image_pil,
                                num_colors=num_colors,
                                palette_colors=None,
                                resize_factor=resize_factor if 'resize_factor' in locals() else 1.0
                            )
                            
                            if masks is not None and len(masks) > 0:
                                # Применяем порог к маскам если задан
                                if 'mask_threshold' in locals():
                                    masks = np.where(masks > mask_threshold, masks, 0)
                                
                                # Применяем сглаживание если включено
                                if 'mask_smoothing' in locals() and mask_smoothing:
                                    kernel_size = smoothing_kernel if 'smoothing_kernel' in locals() else 3
                                    for i in range(len(masks)):
                                        masks[i] = cv2.GaussianBlur(masks[i], 
                                                                   (kernel_size, kernel_size), 0)
                                
                                # Создаем цветные слои из масок
                                color_layers, color_info = create_color_layers_from_masks(
                                    img_cv,
                                    masks,
                                    colors_rgb,
                                    bg_color=bg_color_rgb
                                )
                            else:
                                st.error("Не удалось выполнить разделение с помощью нейронной сети.")
                                color_layers, color_info = [], []
                    
                    # Сохраняем результаты в session state
                    st.session_state.color_layers = color_layers
                    st.session_state.color_info = color_info
                    st.session_state.processed_image = True
                    
                    if color_layers and color_info:
                        st.success(f"✅ Успешно создано {len(color_layers)} цветовых слоев!")
                        
                        # Показываем статистику
                        total_coverage = sum(info['coverage'] for info in color_info)
                        avg_intensity = np.mean([info['intensity'] for info in color_info])
                        
                        st.info(f"""
                        📊 Статистика разделения:
                        - Общее покрытие: {total_coverage:.1f}%
                        - Средняя интенсивность: {avg_intensity:.1f}%
                        - Фоновый цвет: RGB{bg_color_rgb}
                        """)
                    else:
                        st.warning("⚠️ Не удалось создать цветовые слои. Попробуйте изменить параметры.")
                        
                except Exception as e:
                    st.error(f"❌ Ошибка при обработке изображения: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Показываем результаты если они есть
        color_layers = st.session_state.color_layers
        color_info = st.session_state.color_info
        
        if color_layers and color_info and st.session_state.processed_image:
            # Создаем вкладки для каждого слоя
            tab_titles = []
            for i, info in enumerate(color_info):
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    info['color'][2], info['color'][1], info['color'][0]
                )
                coverage = info['coverage']
                tab_titles.append(f"Слой {i+1} ({coverage:.1f}%)")
            
            tabs = st.tabs(tab_titles)
            
            for i, (layer, info) in enumerate(zip(color_layers, color_info)):
                with tabs[i]:
                    col_left, col_right = st.columns([3, 1])
                    
                    with col_left:
                        # Отображаем слой
                        layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                        st.image(layer_rgb, use_column_width=True, 
                                caption=f"Цветовой слой {i+1}")
                        
                        # Показываем маску если доступна
                        if 'mask' in info:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                            
                            # Маска
                            ax1.imshow(info['mask'], cmap='gray')
                            ax1.set_title(f'Маска слоя {i+1}')
                            ax1.axis('off')
                            
                            # Гистограмма маски
                            ax2.hist(info['mask'].flatten(), bins=50, color='blue', alpha=0.7)
                            ax2.set_title('Распределение значений маски')
                            ax2.set_xlabel('Значение маски')
                            ax2.set_ylabel('Количество пикселей')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Кнопки для скачивания
                        col_btn1, col_btn2, col_btn3 = st.columns(3)
                        
                        with col_btn1:
                            # Черно-белая маска
                            if 'mask' in info:
                                bw_mask = create_bw_mask(info['mask'])
                                mask_png = save_image_as_png(bw_mask, f"mask_{i+1}.png")
                                
                                if mask_png:
                                    st.download_button(
                                        label="⬇️ ЧБ маска",
                                        data=mask_png,
                                        file_name=f"layer_{i+1}_mask.png",
                                        mime="image/png",
                                        key=f"download_mask_{i}"
                                    )
                        
                        with col_btn2:
                            # Цветной слой
                            layer_png = save_image_as_png(layer, f"layer_{i+1}.png")
                            if layer_png:
                                st.download_button(
                                    label="⬇️ Цветной слой",
                                    data=layer_png,
                                    file_name=f"layer_{i+1}_color.png",
                                    mime="image/png",
                                    key=f"download_color_{i}"
                                )
                        
                        with col_btn3:
                            # Альфа-слой (градации серого)
                            if 'mask' in info:
                                alpha_layer = (info['mask'] * 255).astype(np.uint8)
                                alpha_png = save_image_as_png(alpha_layer, f"alpha_{i+1}.png")
                                
                                if alpha_png:
                                    st.download_button(
                                        label="⬇️ Альфа-канал",
                                        data=alpha_png,
                                        file_name=f"layer_{i+1}_alpha.png",
                                        mime="image/png",
                                        key=f"download_alpha_{i}"
                                    )
                    
                    with col_right:
                        # Информация о цвете
                        hex_color = "#{:02x}{:02x}{:02x}".format(
                            info['color'][2], info['color'][1], info['color'][0]
                        )
                        target_hex = "#{:02x}{:02x}{:02x}".format(*info['target_color']) if 'target_color' in info else hex_color
                        
                        # Вычисляем сходство с целевым цветом
                        if 'target_color' in info:
                            similarity = calculate_color_similarity(
                                info['color'], 
                                info['target_color']
                            )
                        else:
                            similarity = 100
                        
                        st.markdown(f"""
                        <div style='padding: 15px; background-color: #f8f9fa; border-radius: 10px;'>
                            <div style='margin-bottom: 15px;'>
                                <strong style='font-size: 1.1em;'>Фактический цвет:</strong><br>
                                <div style='display: flex; align-items: center; margin: 10px 0;'>
                                    <div class='color-chip' style='background-color: {hex_color};'></div>
                                    <div>
                                        <strong>{hex_color}</strong><br>
                                        <span style='color: #666;'>RGB{info['color'][::-1]}</span>
                                    </div>
                                </div>
                            </div>
                            
                            <div style='margin-bottom: 15px;'>
                                <strong style='font-size: 1.1em;'>Целевой цвет:</strong><br>
                                <div style='display: flex; align-items: center; margin: 10px 0;'>
                                    <div class='color-chip' style='background-color: {target_hex};'></div>
                                    <div>
                                        <strong>{target_hex}</strong><br>
                                        <span style='color: #666;'>RGB{info.get('target_color', info['color'][::-1])}</span>
                                    </div>
                                </div>
                            </div>
                            
                            <div style='border-top: 1px solid #ddd; padding-top: 10px;'>
                                <strong>Сходство:</strong> {similarity:.1f}%<br>
                                <strong>Покрытие:</strong> {info['coverage']:.1f}%<br>
                                <strong>Интенсивность:</strong> {info['intensity']:.1f}%<br>
                                <strong>Размер:</strong> {layer.shape[1]} × {layer.shape[0]} px
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # ==================== КОМБИНИРОВАННЫЙ ПРЕДПРОСМОТР ====================
            
            st.markdown("---")
            st.markdown("<h3 class='sub-header'>👁️ Комбинированный предпросмотр</h3>", unsafe_allow_html=True)
            
            # Настройки порядка слоев
            with st.expander("⚙️ Управление слоями", expanded=True):
                # Инициализация
                if 'layer_order' not in st.session_state or len(st.session_state.layer_order) != len(color_layers):
                    st.session_state.layer_order = list(range(len(color_layers)))
                if 'layer_visibility' not in st.session_state or len(st.session_state.layer_visibility) != len(color_layers):
                    st.session_state.layer_visibility = [True] * len(color_layers)
                if 'layer_opacity' not in st.session_state or len(st.session_state.layer_opacity) != len(color_layers):
                    st.session_state.layer_opacity = [1.0] * len(color_layers)
                
                # Таблица управления слоями
                for i in range(len(color_layers)):
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
                    
                    with col1:
                        # Порядок
                        order = st.number_input(
                            "Поз.",
                            min_value=1,
                            max_value=len(color_layers),
                            value=i+1,
                            key=f"order_{i}",
                            label_visibility="collapsed"
                        )
                        st.session_state.layer_order[i] = order - 1
                    
                    with col2:
                        # Видимость
                        visible = st.checkbox(
                            "Вкл",
                            value=st.session_state.layer_visibility[i],
                            key=f"visible_{i}",
                            label_visibility="collapsed"
                        )
                        st.session_state.layer_visibility[i] = visible
                    
                    with col3:
                        # Прозрачность
                        opacity = st.slider(
                            "Непр.",
                            min_value=0.0,
                            max_value=1.0,
                            value=st.session_state.layer_opacity[i],
                            key=f"opacity_{i}",
                            label_visibility="collapsed"
                        )
                        st.session_state.layer_opacity[i] = opacity
                    
                    with col4:
                        # Информация о слое
                        hex_color = "#{:02x}{:02x}{:02x}".format(
                            color_info[i]['color'][2], color_info[i]['color'][1], color_info[i]['color'][0]
                        )
                        bg_color = '#e8f5e9' if visible else '#f5f5f5'
                        st.markdown(f"""
                        <div style='padding: 8px; background-color: {bg_color}; border-radius: 5px;'>
                            <div style='display: flex; align-items: center;'>
                                <div style='width: 20px; height: 20px; background-color: {hex_color}; 
                                         border: 1px solid #000; border-radius: 3px; margin-right: 10px;'></div>
                                <div style='flex-grow: 1;'>
                                    <strong>Слой {i+1}</strong> • {hex_color} • {color_info[i]['coverage']:.1f}%
                                </div>
                                <div style='font-size: 0.8em; color: #666;'>
                                    Непр: {opacity:.1f}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Создание комбинированного изображения
            combined = np.full_like(img_cv, bg_color_bgr, dtype=np.uint8)
            
            # Сортируем слои по порядку (от нижнего к верхнему)
            sorted_indices = sorted(range(len(st.session_state.layer_order)), 
                                   key=lambda x: st.session_state.layer_order[x])
            
            # Применяем слои с учетом видимости и прозрачности
            for idx in sorted_indices:
                if st.session_state.layer_visibility[idx]:
                    layer = color_layers[idx]
                    opacity = st.session_state.layer_opacity[idx]
                    
                    # Изменяем размер если нужно
                    if layer.shape != combined.shape:
                        layer = resize_image_to_match(layer, combined.shape)
                    
                    # Создаем маску для текущего слоя
                    if 'mask' in color_info[idx]:
                        mask = color_info[idx]['mask']
                        if mask.shape != combined.shape[:2]:
                            mask = cv2.resize(mask, (combined.shape[1], combined.shape[0]))
                    else:
                        # Создаем маску из слоя (не фон)
                        mask = np.any(layer != bg_color_bgr, axis=2).astype(float)
                    
                    # Применяем прозрачность к маске
                    mask = mask * opacity
                    
                    # Смешиваем слой с комбинированным изображением
                    for c in range(3):
                        combined[:, :, c] = (
                            layer[:, :, c] * mask + 
                            combined[:, :, c] * (1 - mask)
                        ).astype(np.uint8)
            
            # Сохраняем комбинированный превью
            st.session_state.combined_preview = combined
            
            # Отображаем комбинированное изображение
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            
            visible_count = sum(st.session_state.layer_visibility)
            st.image(combined_rgb, 
                    caption=f"Комбинированный предпросмотр ({visible_count}/{len(color_layers)} слоев)", 
                    use_column_width=True)
            
            # Кнопки для скачивания комбинированного изображения
            col_comb1, col_comb2 = st.columns(2)
            
            with col_comb1:
                # Комбинированная маска
                combined_mask = np.zeros((combined.shape[0], combined.shape[1]), dtype=np.uint8)
                
                for idx in sorted_indices:
                    if st.session_state.layer_visibility[idx]:
                        if 'mask' in color_info[idx]:
                            mask = color_info[idx]['mask']
                            if mask.shape != combined_mask.shape:
                                mask = cv2.resize(mask, (combined_mask.shape[1], combined_mask.shape[0]))
                            
                            # Применяем порог
                            mask_binary = (mask > 0.1).astype(np.uint8) * 255
                            combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
                
                combined_mask_png = save_image_as_png(combined_mask, "combined_mask.png")
                if combined_mask_png:
                    st.download_button(
                        label="⬇️ Комбинированная маска",
                        data=combined_mask_png,
                        file_name="combined_mask.png",
                        mime="image/png",
                        key="download_combined_mask"
                    )
            
            with col_comb2:
                # Комбинированное цветное изображение
                combined_color_png = save_image_as_png(combined, "combined_preview.png")
                if combined_color_png:
                    st.download_button(
                        label="⬇️ Цветной предпросмотр",
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
                    with tempfile.TemporaryDirectory() as tmpdir:
                        all_files = []
                        
                        # Сохраняем все слои
                        for i, (layer, info) in enumerate(zip(color_layers, color_info)):
                            if st.session_state.layer_visibility[i]:
                                # Черно-белая маска
                                if 'mask' in info:
                                    bw_mask = create_bw_mask(info['mask'])
                                    mask_png = save_image_as_png(bw_mask)
                                    if mask_png:
                                        mask_path = os.path.join(tmpdir, f"layer_{i+1}_mask.png")
                                        with open(mask_path, 'wb') as f:
                                            f.write(mask_png)
                                        all_files.append(("mask", mask_path))
                                
                                # Цветной слой
                                layer_png = save_image_as_png(layer)
                                if layer_png:
                                    layer_path = os.path.join(tmpdir, f"layer_{i+1}_color.png")
                                    with open(layer_path, 'wb') as f:
                                        f.write(layer_png)
                                    all_files.append(("color", layer_path))
                                
                                # Альфа-канал
                                if 'mask' in info:
                                    alpha_layer = (info['mask'] * 255).astype(np.uint8)
                                    alpha_png = save_image_as_png(alpha_layer)
                                    if alpha_png:
                                        alpha_path = os.path.join(tmpdir, f"layer_{i+1}_alpha.png")
                                        with open(alpha_path, 'wb') as f:
                                            f.write(alpha_png)
                                        all_files.append(("alpha", alpha_path))
                        
                        # Сохраняем комбинированные изображения
                        if combined_mask_png:
                            combined_mask_path = os.path.join(tmpdir, "combined_mask.png")
                            with open(combined_mask_path, 'wb') as f:
                                f.write(combined_mask_png)
                            all_files.append(("combined", combined_mask_path))
                        
                        if combined_color_png:
                            combined_color_path = os.path.join(tmpdir, "combined_preview.png")
                            with open(combined_color_path, 'wb') as f:
                                f.write(combined_color_png)
                            all_files.append(("combined", combined_color_path))
                        
                        # Создаем README файл
                        readme_content = f"""# ColorSep Pro - Экспортированные слои

Дата создания: {st.session_state.get('processing_time', 'Неизвестно')}
Метод: {selected_method}
Количество слоев: {len(color_layers)}
Цвет фона: {bg_color}

## Информация о слоях:
"""
                        
                        for i, info in enumerate(color_info):
                            hex_color = "#{:02x}{:02x}{:02x}".format(
                                info['color'][2], info['color'][1], info['color'][0]
                            )
                            readme_content += f"""
Слой {i+1}:
- Цвет: {hex_color} (RGB{info['color'][::-1]})
- Покрытие: {info['coverage']:.1f}%
- Интенсивность: {info['intensity']:.1f}%
- Видим: {'Да' if st.session_state.layer_visibility[i] else 'Нет'}
- Непрозрачность: {st.session_state.layer_opacity[i]:.1f}

"""
                        
                        readme_path = os.path.join(tmpdir, "README.txt")
                        with open(readme_path, 'w', encoding='utf-8') as f:
                            f.write(readme_content)
                        all_files.append(("info", readme_path))
                        
                        # Создаем ZIP архив
                        zip_path = os.path.join(tmpdir, "color_layers.zip")
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for file_type, file_path in all_files:
                                zipf.write(file_path, os.path.basename(file_path))
                        
                        # Читаем ZIP файл
                        with open(zip_path, "rb") as f:
                            zip_data = f.read()
                        
                        # Предоставляем для скачивания
                        st.download_button(
                            label="⬇️ Скачать ZIP архив",
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
                <li>Точные маски с мягкими краями</li>
            </ul>
            <p><strong>Идеально для:</strong> Фотографии, градиенты, сложные текстуры, художественные работы</p>
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

# ==================== СОВЕТЫ И РЕКОМЕНДАЦИИ ====================

st.markdown("---")
st.markdown("<h3 class='sub-header'>💡 Советы для лучших результатов</h3>", unsafe_allow_html=True)

col_tip1, col_tip2, col_tip3 = st.columns(3)

with col_tip1:
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; height: 100%;">
        <h5>🎨 Выбор метода</h5>
        <ul style="margin-bottom: 0;">
            <li><strong>K-means:</strong> Для логотипов и графики</li>
            <li><strong>Нейронная сеть:</strong> Для фотографий и градиентов</li>
            <li>Начните с 5-6 цветов для сложных изображений</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_tip2:
    st.markdown("""
    <div style="background-color: #f3e5f5; padding: 15px; border-radius: 10px; height: 100%;">
        <h5>⚡ Настройки нейронной сети</h5>
        <ul style="margin-bottom: 0;">
            <li>Масштаб 1.0 для большинства изображений</li>
            <li>Порог маски 0.1-0.2 для четких границ</li>
            <li>Включите сглаживание для плавных переходов</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_tip3:
    st.markdown("""
    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; height: 100%;">
        <h5>📊 Анализ результатов</h5>
        <ul style="margin-bottom: 0;">
            <li>Проверьте покрытие каждого слоя</li>
            <li>Используйте прозрачность для смешивания</li>
            <li>Сравните целевые и фактические цвета</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================== ФУТЕР ====================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 30px; background-color: #f8f9fa; border-radius: 10px;">
    <h4>🎨 ColorSep Pro v2.0</h4>
    <p>Профессиональный инструмент для разделения цветов с нейронными сетями</p>
    <p style="font-size: 0.9em;">Поддерживаемые форматы: JPG, PNG, BMP, TIFF | Максимальный размер: 50MB</p>
    <p style="font-size: 0.9em;">Все файлы экспортируются в формате PNG с сохранением прозрачности</p>
</div>
""", unsafe_allow_html=True)

# ==================== ПРОВЕРКА ЗАВИСИМОСТЕЙ ====================

try:
    with st.sidebar.expander("ℹ️ Информация о системе", expanded=False):
        st.write(f"**OpenCV:** {cv2.__version__}")
        st.write(f"**PyTorch:** {torch.__version__}")
        st.write(f"**CUDA доступен:** {'✅ Да' if torch.cuda.is_available() else '❌ Нет'}")
        st.write(f"**Streamlit:** {st.__version__}")
        
        # Проверка памяти
        if torch.cuda.is_available():
            st.write(f"**GPU Память:** {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Статус модели
        if model_available:
            st.success("✅ Модель нейронной сети загружена")
        else:
            st.warning("⚠️ Модель нейронной сети не найдена")
            
except Exception as e:
    st.sidebar.error(f"Ошибка проверки системы: {e}")

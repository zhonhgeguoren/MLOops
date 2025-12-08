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
    st.session_state.selected_method = "K-средних кластеризация (улучшенный)"

if 'combined_preview' not in st.session_state:
    st.session_state.combined_preview = None

# ==================== ФУНКЦИИ ЦВЕТОВОГО АНАЛИЗА ====================

def get_dominant_colors_kmeans(img_cv, n_colors=5, bg_color=(255, 255, 255), compactness=1.0):
    """
    Улучшенный K-means для точного определения доминирующих цветов
    Возвращает цвета в формате RGB
    """
    # Преобразуем изображение в формат для K-means
    pixels = img_cv.reshape(-1, 3)
    
    # Удаляем пиксели фона
    if bg_color is not None:
        bg_color_np = np.array(bg_color)
        # Используем порог для определения фона
        distance = np.linalg.norm(pixels - bg_color_np, axis=1)
        non_bg_mask = distance > 20  # Пороговое значение
        pixels_for_clustering = pixels[non_bg_mask]
    else:
        pixels_for_clustering = pixels
    
    # Если после удаления фона не осталось пикселей
    if len(pixels_for_clustering) == 0:
        # Возвращаем равномерно распределенные цвета
        colors = []
        for i in range(n_colors):
            r = int(255 * i / (n_colors - 1))
            g = int(255 * (n_colors - i - 1) / (n_colors - 1))
            b = int(255 * (i % 3) / 3)
            colors.append([r, g, b])
        return np.array(colors)
    
    try:
        # Настраиваем параметры K-means для лучшей компактности
        kmeans = KMeans(
            n_clusters=n_colors,
            random_state=42,
            n_init=10,
            max_iter=300,
            tol=1e-4
        )
        
        # Применяем веса для компактности
        if compactness != 1.0:
            # Увеличиваем количество итераций для лучшей сходимости
            kmeans.set_params(max_iter=500)
        
        labels = kmeans.fit_predict(pixels_for_clustering)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Сортируем цвета по насыщенности/яркости для лучшего восприятия
        colors_hsv = []
        for color in colors:
            r, g, b = color / 255.0
            cmax = max(r, g, b)
            cmin = min(r, g, b)
            delta = cmax - cmin
            
            # Вычисляем насыщенность
            if cmax == 0:
                saturation = 0
            else:
                saturation = delta / cmax
            
            # Вычисляем яркость
            value = cmax
            
            colors_hsv.append((saturation, value))
        
        # Сортируем по насыщенности (сначала более насыщенные цвета)
        sorted_indices = np.argsort([-hsv[0] for hsv in colors_hsv])
        colors = colors[sorted_indices]
        
        return colors
    
    except Exception as e:
        st.error(f"Ошибка в K-means: {str(e)}")
        # Возвращаем запасные цвета
        colors = []
        for i in range(n_colors):
            r = int(255 * i / (n_colors - 1))
            g = int(255 * (n_colors - i - 1) / (n_colors - 1))
            b = int(255 * (i % 3) / 3)
            colors.append([r, g, b])
        return np.array(colors)

def enhanced_kmeans_color_separation(img_cv, n_colors=5, bg_color=(255, 255, 255), 
                                    compactness=1.0, noise_reduction=2,
                                    apply_smoothing=True, smoothing_amount=3,
                                    apply_sharpening=False, sharpening_amount=1.0):
    """
    Улучшенный метод K-means с настройками компактности и постобработкой
    """
    if n_colors < 2 or n_colors > 15:
        st.error(f"Количество цветов должно быть от 2 до 15. Получено: {n_colors}")
        return [], []
    
    try:
        # Получаем доминирующие цвета с улучшенным K-means
        dominant_colors = get_dominant_colors_kmeans(img_cv, n_colors, bg_color, compactness)
        
        # Преобразуем цвета в BGR для OpenCV
        dominant_colors_bgr = []
        for color in dominant_colors:
            bgr_color = (int(color[2]), int(color[1]), int(color[0]))
            dominant_colors_bgr.append(bgr_color)
        
        # Создаем маску для каждого цвета
        color_layers = []
        color_info = []
        
        # Применяем шумоподавление к исходному изображению
        if noise_reduction > 0:
            img_processed = cv2.medianBlur(img_cv, noise_reduction * 2 + 1)
        else:
            img_processed = img_cv.copy()
        
        # Создаем маску фона
        bg_color_np = np.array(bg_color)
        bg_mask = np.all(img_processed == bg_color_np, axis=2)
        
        # Преобразуем изображение в формат для кластеризации
        pixels = img_processed.reshape(-1, 3)
        
        # Удаляем пиксели фона
        non_bg_mask = ~bg_mask.reshape(-1)
        pixels_for_clustering = pixels[non_bg_mask]
        
        if len(pixels_for_clustering) == 0:
            st.warning("Изображение состоит только из фона")
            return [], []
        
        # Вычисляем расстояния до каждого доминирующего цвета
        distances = np.zeros((len(pixels_for_clustering), len(dominant_colors_bgr)))
        for i, color in enumerate(dominant_colors_bgr):
            color_np = np.array(color)
            distances[:, i] = np.linalg.norm(pixels_for_clustering - color_np, axis=1)
        
        # Назначаем каждый пиксель ближайшему цвету
        labels = np.argmin(distances, axis=1)
        
        # Создаем полную маску меток
        full_labels = np.zeros(img_cv.shape[0] * img_cv.shape[1], dtype=int) - 1
        full_labels[non_bg_mask] = labels
        
        # Создаем слои для каждого цвета
        for i, color_bgr in enumerate(dominant_colors_bgr):
            # Создаем маску для текущего кластера
            mask = (full_labels == i).reshape(img_cv.shape[0], img_cv.shape[1])
            
            # Применяем сглаживание к маске если нужно
            if apply_smoothing and smoothing_amount > 0:
                mask_float = mask.astype(float)
                kernel_size = smoothing_amount * 2 + 1
                mask_smoothed = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), 0)
                mask = (mask_smoothed > 0.5).astype(bool)
            
            # Создаем слой
            layer = np.full_like(img_cv, bg_color)
            
            # Если есть пиксели в маске, применяем цвет
            if np.any(mask):
                for c in range(3):
                    layer[:, :, c] = np.where(mask, color_bgr[c], bg_color[c])
            
            # Применяем увеличение резкости если нужно
            if apply_sharpening and sharpening_amount > 0:
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]]) * sharpening_amount
                layer = cv2.filter2D(layer, -1, kernel)
                layer = np.clip(layer, 0, 255)
            
            # Вычисляем процент покрытия
            coverage_percentage = (np.sum(mask) / mask.size) * 100
            
            # Получаем оригинальный цвет в RGB для информации
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            
            color_layers.append(layer)
            color_info.append({
                'color': color_bgr,
                'percentage': coverage_percentage,
                'rgb_color': color_rgb,
                'method': 'enhanced_kmeans',
                'compactness': compactness
            })
        
        return color_layers, color_info
    
    except Exception as e:
        st.error(f"❌ Ошибка в улучшенном методе K-means: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return [], []

def exact_color_separation(img_cv, max_colors=10, bg_color=(255, 255, 255)):
    """
    Точное извлечение цветов - создает слой для каждого уникального цвета
    """
    try:
        # Убираем фон
        bg_color_np = np.array(bg_color)
        non_bg_mask = ~np.all(img_cv == bg_color_np, axis=2)
        
        if not np.any(non_bg_mask):
            return [], []
        
        # Получаем уникальные цвета (без фона)
        colors_flat = img_cv[non_bg_mask].reshape(-1, 3)
        
        # Находим уникальные цвета
        unique_colors, counts = np.unique(colors_flat, axis=0, return_counts=True)
        
        # Сортируем по частоте
        sorted_indices = np.argsort(counts)[::-1]
        unique_colors = unique_colors[sorted_indices]
        counts = counts[sorted_indices]
        
        # Ограничиваем количество цветов
        num_colors = min(max_colors, len(unique_colors))
        unique_colors = unique_colors[:num_colors]
        counts = counts[:num_colors]
        
        # Создаем слои
        color_layers = []
        color_info = []
        
        total_pixels = np.sum(counts)
        
        for i, color in enumerate(unique_colors):
            # Создаем маску для этого цвета
            color_np = np.array(color)
            mask = np.all(img_cv == color_np, axis=2)
            
            # Создаем слой
            layer = np.full_like(img_cv, bg_color)
            layer[mask] = color
            
            # Процент покрытия
            coverage_percentage = (np.sum(mask) / mask.size) * 100
            
            # Цвет в BGR и RGB
            color_bgr = (int(color[0]), int(color[1]), int(color[2]))
            color_rgb = (int(color[2]), int(color[1]), int(color[0]))
            
            color_layers.append(layer)
            color_info.append({
                'color': color_bgr,
                'percentage': coverage_percentage,
                'rgb_color': color_rgb,
                'method': 'exact',
                'count': int(counts[i])
            })
        
        return color_layers, color_info
    
    except Exception as e:
        st.error(f"❌ Ошибка в точном извлечении цветов: {str(e)}")
        return [], []

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def convert_to_png(image_array, filename):
    """Конвертирует массив изображения в формат PNG"""
    try:
        # Если изображение в формате BGR, конвертируем в RGB
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_array
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_rgb)
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

def get_color_from_code(color_code):
    """Преобразует HEX или RGB код в цвет BGR"""
    if isinstance(color_code, str) and color_code.startswith('#'):
        # HEX код
        hex_color = color_code.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)  # BGR формат
    elif isinstance(color_code, tuple) and len(color_code) == 3:
        # RGB tuple
        return (color_code[2], color_code[1], color_code[0])  # BGR формат
    else:
        return (255, 255, 255)  # Белый по умолчанию

# ==================== НАСТРОЙКА СТРАНИЦЫ ====================

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
    .compactness-badge {
        background: linear-gradient(45deg, #4CAF50, #2196F3);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        display: inline-block;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown("<h1 class='main-header'>ColorSep Pro: Профессиональное разделение цветов</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text' style='text-align: center;'>Загрузите изображение и извлеките цветовые слои для печати и дизайна</p>", unsafe_allow_html=True)

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
        methods = [
            "K-средних кластеризация (улучшенный)",
            "Точное извлечение цветов",
            "Пантон цвета (TPX/TPG)"
        ]
        
        selected_method = st.selectbox("Метод разделения", methods, 
                                      label_visibility="collapsed")
        st.session_state.selected_method = selected_method
        
        # Количество цветов
        st.markdown("<h4>🌈 Количество цветов</h4>", unsafe_allow_html=True)
        num_colors = st.slider("Количество цветов для извлечения", 2, 15, 5, 
                              help="Выберите количество цветов для извлечения из изображения",
                              label_visibility="collapsed")
        
        # Цвет фона
        st.markdown("<h4>🎨 Цвет фона</h4>", unsafe_allow_html=True)
        bg_color = st.color_picker("Цвет фона для слоев", "#FFFFFF", 
                                  label_visibility="collapsed")
        bg_color_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Настройки для улучшенного K-means
        if selected_method == "K-средних кластеризация (улучшенный)":
            st.markdown("<h4>⚡ Настройки K-means</h4>", unsafe_allow_html=True)
            
            compactness = st.slider("Компактность цветов", 0.1, 3.0, 1.0, 0.1,
                                  help="Высокая компактность: четкие границы между цветами\nНизкая компактность: более плавные переходы",
                                  label_visibility="collapsed")
            
            noise_reduction = st.slider("Уменьшение шума", 0, 5, 1,
                                       help="Уменьшает шум перед обработкой",
                                       label_visibility="collapsed")
            
            # Дополнительные опции
            with st.expander("🛠️ Дополнительные настройки", expanded=False):
                apply_smoothing = st.checkbox("Сглаживание границ", True,
                                             help="Сглаживает границы между цветами")
                if apply_smoothing:
                    smoothing_amount = st.slider("Степень сглаживания", 1, 10, 3,
                                                label_visibility="collapsed")
                
                apply_sharpening = st.checkbox("Увеличение резкости", False,
                                              help="Увеличивает резкость границ")
                if apply_sharpening:
                    sharpening_amount = st.slider("Степень резкости", 0.1, 3.0, 1.0, 0.1,
                                                 label_visibility="collapsed")
        
        # Настройки для точного извлечения
        elif selected_method == "Точное извлечение цветов":
            st.markdown("<h4>🎯 Точное извлечение</h4>", unsafe_allow_html=True)
            max_colors = st.slider("Максимальное количество цветов", 5, 50, 20,
                                  help="Извлекает все уникальные цвета до указанного предела",
                                  label_visibility="collapsed")
            st.info("⚠️ Этот метод создает отдельный слой для каждого уникального цвета. Может создавать много слоев для сложных изображений.")
        
        # Настройки для Pantone
        elif selected_method == "Пантон цвета (TPX/TPG)":
            st.markdown("<h4>🎨 Пантон цвета</h4>", unsafe_allow_html=True)
            pantone_code_type = st.radio("Тип кода Pantone", ["TPX", "TPG"], horizontal=True)
            
            # Получаем доступные коды Pantone
            try:
                pantone_codes = pantone.get_all_pantone_codes()
                if pantone_code_type == "TPX":
                    available_codes = pantone_codes.get('TPX', [])
                else:
                    available_codes = pantone_codes.get('TPG', [])
                
                if available_codes:
                    selected_pantone = st.selectbox("Выберите цвет Pantone", available_codes)
                    st.success(f"Выбран: {selected_pantone}")
                else:
                    st.warning("Коды Pantone не найдены. Убедитесь, что pantone_colors.py настроен правильно.")
            except:
                st.warning("Модуль pantone_colors не настроен. Используйте другие методы.")

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
        "K-средних кластеризация (улучшенный)": "Лучше всего подходит для изображений с четкими цветовыми областями",
        "Точное извлечение цветов": "Идеально для векторной графики и логотипов",
        "Пантон цвета (TPX/TPG)": "Профессиональные цвета для текстиля и полиграфии"
    }
    
    st.markdown(f"""
    <div class="method-card">
        <h4>🎯 Выбранный метод: <strong>{selected_method}</strong></h4>
        <p>{method_descriptions.get(selected_method, '')}</p>
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
            
            # Анализ цветов изображения
            st.write("**Анализ цветов:**")
            img_array = np.array(image)
            if img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Подсчет уникальных цветов
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
            st.write(f"Уникальных цветов: {unique_colors}")
            
            # Доминирующие цвета (простые)
            pixels = img_array.reshape(-1, 3)
            unique, counts = np.unique(pixels, axis=0, return_counts=True)
            top_colors = unique[np.argsort(counts)[-5:]][::-1]
            
            st.write("Топ 5 цветов:")
            for i, color in enumerate(top_colors):
                hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
                st.markdown(f'<span style="display: inline-block; width: 20px; height: 20px; background-color: {hex_color}; border: 1px solid #000; margin-right: 10px;"></span> {hex_color}', 
                          unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3 class='sub-header'>🎨 Разделенные цветовые слои</h3>", unsafe_allow_html=True)
        
        # Кнопка для запуска обработки
        if st.button("🚀 Начать разделение цветов", type="primary", use_container_width=True):
            with st.spinner("🔄 Обработка изображения... Пожалуйста, подождите."):
                try:
                    if selected_method == "K-средних кластеризация (улучшенный)":
                        # Используем улучшенный K-means
                        color_layers, color_info = enhanced_kmeans_color_separation(
                            img_cv, 
                            n_colors=num_colors,
                            bg_color=bg_color_rgb,
                            compactness=compactness if 'compactness' in locals() else 1.0,
                            noise_reduction=noise_reduction if 'noise_reduction' in locals() else 0,
                            apply_smoothing=apply_smoothing if 'apply_smoothing' in locals() else False,
                            smoothing_amount=smoothing_amount if 'smoothing_amount' in locals() else 0,
                            apply_sharpening=apply_sharpening if 'apply_sharpening' in locals() else False,
                            sharpening_amount=sharpening_amount if 'sharpening_amount' in locals() else 0
                        )
                    
                    elif selected_method == "Точное извлечение цветов":
                        # Используем точное извлечение
                        color_layers, color_info = exact_color_separation(
                            img_cv,
                            max_colors=max_colors if 'max_colors' in locals() else 10,
                            bg_color=bg_color_rgb
                        )
                    
                    elif selected_method == "Пантон цвета (TPX/TPG)":
                        # Используем Pantone tab
                        pantone_result = pantone_extraction_tab(image, num_colors, bg_color_rgb)
                        if pantone_result:
                            color_layers, color_info = pantone_result
                        else:
                            st.warning("Используем улучшенный K-means как запасной вариант")
                            color_layers, color_info = enhanced_kmeans_color_separation(
                                img_cv, 
                                n_colors=num_colors,
                                bg_color=bg_color_rgb
                            )
                    
                    # Сохраняем результаты в session state
                    st.session_state.color_layers = color_layers
                    st.session_state.color_info = color_info
                    
                    if color_layers and color_info:
                        st.success(f"✅ Успешно создано {len(color_layers)} цветовых слоев!")
                        
                        # Показываем статистику
                        total_coverage = sum(info['percentage'] for info in color_info)
                        avg_coverage = total_coverage / len(color_info)
                        
                        st.info(f"""
                        **Статистика разделения:**
                        - Всего слоев: {len(color_layers)}
                        - Среднее покрытие: {avg_coverage:.1f}%
                        - Общее покрытие: {total_coverage:.1f}%
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
        
        if color_layers and color_info:
            # Создаем вкладки для каждого слоя
            tabs = st.tabs([f"Слой {i+1}" for i in range(len(color_layers))])
            
            for i, (layer, info) in enumerate(zip(color_layers, color_info)):
                with tabs[i]:
                    col_left, col_right = st.columns([3, 1])
                    
                    with col_left:
                        # Конвертация слоя из BGR в RGB для отображения
                        layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                        st.image(layer_rgb, use_column_width=True)
                        
                        # Кнопки для скачивания
                        col_btn1, col_btn2 = st.columns(2)
                        
                        with col_btn1:
                            # Черно-белая маска
                            bw_mask = create_bw_mask(layer, bg_color_rgb)
                            png_data = save_bw_mask_as_png(bw_mask, f"mask_{i+1}")
                            
                            if png_data:
                                hex_color = "{:02x}{:02x}{:02x}".format(
                                    info['color'][2], info['color'][1], info['color'][0]
                                )
                                
                                st.download_button(
                                    label="⬇️ Скачать ЧБ маску",
                                    data=png_data,
                                    file_name=f"layer_{i+1}_mask.png",
                                    mime="image/png",
                                    key=f"download_mask_{i}"
                                )
                        
                        with col_btn2:
                            # Цветной слой
                            color_png_data = convert_to_png(layer_rgb, f"layer_{i+1}")
                            if color_png_data:
                                hex_color = "{:02x}{:02x}{:02x}".format(
                                    info['color'][2], info['color'][1], info['color'][0]
                                )
                                
                                st.download_button(
                                    label="⬇️ Скачать цветной слой",
                                    data=color_png_data,
                                    file_name=f"layer_{i+1}_color.png",
                                    mime="image/png",
                                    key=f"download_color_{i}"
                                )
                    
                    with col_right:
                        # Информация о цвете
                        hex_color = "#{:02x}{:02x}{:02x}".format(
                            info['color'][2], info['color'][1], info['color'][0]
                        )
                        
                        # Метод разделения
                        method_badge = ""
                        if 'method' in info:
                            if info['method'] == 'enhanced_kmeans':
                                method_badge = "<span class='compactness-badge'>Улучшенный K-means</span>"
                            elif info['method'] == 'exact':
                                method_badge = "<span class='compactness-badge' style='background: linear-gradient(45deg, #FF9800, #F44336);'>Точный</span>"
                        
                        st.markdown(f"""
                        <div style='padding: 15px; background-color: #f8f9fa; border-radius: 10px;'>
                            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                                <div class='color-chip' style='background-color: {hex_color};'></div>
                                <div>
                                    <strong style='font-size: 1.2em;'>{hex_color}</strong><br>
                                    <span style='color: #666; font-size: 0.9em;'>Цвет слоя {method_badge}</span>
                                </div>
                            </div>
                            <div style='margin-bottom: 10px;'>
                                <strong>RGB:</strong> {info.get('rgb_color', info['color'][::-1])}<br>
                                <strong>Покрытие:</strong> {info['percentage']:.1f}%<br>
                                <strong>Размер:</strong> {layer.shape[1]} × {layer.shape[0]}<br>
                                <strong>Метод:</strong> {info.get('method', 'N/A')}
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
                
                # Настройки для каждого слоя
                for i in range(len(color_layers)):
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
                        hex_color = "#{:02x}{:02x}{:02x}".format(
                            color_info[i]['color'][2], color_info[i]['color'][1], color_info[i]['color'][0]
                        )
                        
                        # Дополнительная информация
                        extra_info = ""
                        if 'count' in color_info[i]:
                            extra_info = f"<br><span style='font-size: 0.8em; color: #666;'>Пикселей: {color_info[i]['count']:,}</span>"
                        
                        st.markdown(f"""
                        <div style='display: flex; align-items: center; padding: 8px; background-color: {'#e8f5e9' if visibility else '#f5f5f5'}; border-radius: 5px;'>
                            <div style='width: 25px; height: 25px; background-color: {hex_color}; border: 1px solid #000; border-radius: 4px; margin-right: 10px;'></div>
                            <div>
                                <div><strong>Слой {i+1}</strong></div>
                                <div style='font-size: 0.8em; color: #666;'>{hex_color} • {color_info[i]['percentage']:.1f}%{extra_info}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Создание комбинированного изображения
            combined = np.zeros_like(img_cv, dtype=np.uint8)
            
            # Сортируем индексы по порядку (от нижнего к верхнему)
            sorted_indices = sorted(range(len(st.session_state.layer_order)), 
                                   key=lambda x: st.session_state.layer_order[x])
            
            # Применяем слои в правильном порядке
            for idx in sorted_indices:
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
                                # Черно-белая маска
                                bw_mask = create_bw_mask(layer, bg_color_rgb)
                                mask_png = save_bw_mask_as_png(bw_mask, f"mask_{i+1}")
                                
                                if mask_png:
                                    mask_path = os.path.join(tmpdirname, f"layer_{i+1}_mask.png")
                                    with open(mask_path, 'wb') as f:
                                        f.write(mask_png)
                                    all_files.append(mask_path)
                                
                                # Цветной слой
                                layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                                color_png = convert_to_png(layer_rgb, f"layer_{i+1}")
                                
                                if color_png:
                                    color_path = os.path.join(tmpdirname, f"layer_{i+1}_color.png")
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
                            hex_color = "#{:02x}{:02x}{:02x}".format(
                                info['color'][2], info['color'][1], info['color'][0]
                            )
                            rgb_color = info.get('rgb_color', info['color'][::-1])
                            
                            readme_content += f"- Слой {i+1}: {hex_color}, RGB{rgb_color}, Покрытие: {info['percentage']:.1f}%\n"
                        
                        readme_path = os.path.join(tmpdirname, "README.txt")
                        with open(readme_path, 'w', encoding='utf-8') as f:
                            f.write(readme_content)
                        all_files.append(readme_path)
                        
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
        <h4>🎯 K-средних кластеризация (улучшенный)</h4>
        <p><strong>Описание:</strong> Улучшенный алгоритм K-means с настройкой компактности и постобработкой.</p>
        <p><strong>Преимущества:</strong></p>
        <ul>
            <li>Отличное распознавание цветов</li>
            <li>Настройка компактности для четких или плавных переходов</li>
            <li>Шумоподавление и сглаживание</li>
            <li>Идеально для изображений с четкими цветовыми областями</li>
        </ul>
        <p><strong>Идеально для:</strong> Логотипы, векторная графика, четкие цветовые области</p>
    </div>
    """, unsafe_allow_html=True)

with col_method2:
    st.markdown("""
    <div class="method-card">
        <h4>🎯 Точное извлечение цветов</h4>
        <p><strong>Описание:</strong> Извлекает каждый уникальный цвет как отдельный слой.</p>
        <p><strong>Преимущества:</strong></p>
        <ul>
            <li>Сохранение всех оригинальных цветов</li>
            <li>Идеально для векторных изображений</li>
            <li>Без потерь цветовой информации</li>
        </ul>
        <p><strong>Идеально для:</strong> Векторная графика, логотипы, пиксель-арт</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== ПАНТОН ВКЛАДКА ====================

if model_available:
    st.markdown("---")
    st.markdown("<h2 class='sub-header'>🎨 Пантон цвета</h2>", unsafe_allow_html=True)
    
    # Инициализируем pantone_tab если доступен
    try:
        pantone_extraction_tab(image if 'image' in locals() else None, 
                              num_colors if 'num_colors' in locals() else 5, 
                              bg_color_rgb if 'bg_color_rgb' in locals() else (255, 255, 255))
    except:
        st.info("Модуль pantone_tab требует дополнительной настройки. Убедитесь, что файл pantone_tab.py находится в той же директории.")

# ==================== ФУТЕР ====================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 30px; background-color: #f8f9fa; border-radius: 10px;">
    <h4>🎨 ColorSep Pro</h4>
    <p>Профессиональный инструмент для разделения цветов</p>
    <p style="font-size: 0.9em;">Поддерживаемые форматы: JPG, PNG, BMP, TIFF | Максимальный размер: 50MB</p>
    <p style="font-size: 0.9em;">Все файлы экспортируются в формате PNG для промышленной совместимости</p>
</div>
""", unsafe_allow_html=True)

# ==================== ПРОВЕРКА ЗАВИСИМОСТЕЙ ====================

try:
    # Проверяем основные зависимости
    dependencies_ok = True
    
    # Проверка OpenCV
    cv2_version = cv2.__version__
    
    # Проверка scikit-learn
    from sklearn import __version__ as sklearn_version
    
    # Выводим информацию в sidebar
    with st.sidebar.expander("ℹ️ Информация о системе", expanded=False):
        st.write(f"**OpenCV:** {cv2_version}")
        st.write(f"**scikit-learn:** {sklearn_version}")
        st.write(f"**Streamlit:** {st.__version__}")
        st.write(f"**NumPy:** {np.__version__}")
        st.write(f"**PIL:** {Image.__version__}")
        
except Exception as e:
    st.sidebar.error(f"Ошибка проверки зависимостей: {e}")

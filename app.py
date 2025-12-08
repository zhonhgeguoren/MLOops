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

# Настройка страницы
st.set_page_config(
    page_title="ColorSep - Инструмент разделения цветов для текстиля",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Настройка темы - светлая тема для лучшей видимости текста
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
    }
    .color-chip {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 5px;
        border: 1px solid #000;
    }
    /* Улучшение видимости текста */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, label, .stSelectbox, .stSlider {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    /* Улучшение контраста для меток */
    .stSelectbox label, .stSlider label {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    /* Добавление фона для важных секций */
    .stExpander {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown("<h1 class='main-header'>ColorSep: Инструмент разделения цветов для текстиля</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Загрузите изображение и извлеките различные цветовые слои для текстильной печати</p>", unsafe_allow_html=True)

# Инициализация переменных состояния сессии
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

# Классы для метода Decompose (Fast Soft Color Segmentation)
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

def decompose_fast_soft_color(
    input_image: Image.Image,
    palette: list[tuple] = None,
    guided_filter=True,
    normalize_alpha=True,
    resize_scale_factor=1
) -> list[Image.Image]:
    """
    Функция для разложения изображения на цветовые слои с использованием нейронной сети
    """
    layersRGBA = []
    num_primary_color = 7
    
    # Преобразование изображения PIL в формат для обработки
    if palette is None:
        # Используем доминирующие цвета, если палитра не задана
        palette = get_dominant_colors(input_image, num_primary_color)
    
    palette = np.array(palette)
    test_dataset = _MyDataset(input_image, num_primary_color, palette)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    cpu = torch.device("cpu")
    
    # Загрузка модели
    mask_generator = _MaskGeneratorModel(num_primary_color).to(cpu)
    
    # Загрузка весов модели (предполагается, что файл модели доступен)
    try:
        # Пытаемся загрузить модель из локальной директории
        model_path = Path(__file__).parent / "model" / "mask_generator7.pth"
        mask_generator.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
    except:
        st.warning("Модель Fast Soft Color Segmentation не найдена. Используйте другие методы.")
        return []
    
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
            
            if normalize_alpha:
                processed_alpha_layers = alpha_normalize(processed_alpha_layers)
            
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

# Функция для получения доминирующих цветов (аналог функции из decompose)
def get_dominant_colors(img: Image.Image, num_colors: int) -> list[tuple]:
    """
    Получение доминирующих цветов из изображения
    """
    import numpy as np
    from numpy import linalg as LA
    from collections import deque
    
    class _ColorNode(object):
        def __init__(self):
            self.__mean = None
            self.__cov = None
            self.__class_id = None
            self.__left = None
            self.__right = None
            self.__num_pixel = None
        
        @property
        def mean(self):
            return self.__mean
        
        @mean.setter
        def mean(self, mean):
            self.__mean = mean
        
        @property
        def cov(self):
            return self.__cov
        
        @cov.setter
        def cov(self, cov):
            self.__cov = cov
        
        @property
        def class_id(self):
            return self.__class_id
        
        @class_id.setter
        def class_id(self, class_id):
            self.__class_id = class_id
        
        @property
        def left(self):
            return self.__left
        
        @left.setter
        def left(self, left):
            self.__left = left
        
        @property
        def right(self):
            return self.__right
        
        @right.setter
        def right(self, right):
            self.__right = right
        
        @property
        def num_pixel(self):
            return self.__num_pixel
        
        @num_pixel.setter
        def num_pixel(self, num_pixel):
            self.__num_pixel = num_pixel
    
    def _rgba2rgb(rgba):
        """
        Конвертация RGBA в RGB с белым фоном
        """
        background = (255, 255, 255)
        alpha = rgba[..., -1]
        channels = rgba[..., :-1]
        out = np.empty_like(channels)
        for ichan in range(channels.shape[-1]):
            w = alpha / 255.0
            out[..., ichan] = np.clip(
                w * channels[..., ichan] + (1 - w) * background[ichan], a_min=0, a_max=255
            )
        out.astype(np.uint8)
        return out
    
    def _find_dominant_colors(img_colors, count):
        """
        Нахождение доминирующих цветов
        """
        colors = img_colors / 255.0
        if len(colors.shape) == 3 and colors.shape[-1] == 3:
            colors = colors.reshape((-1, 3))
        classes = np.ones(colors.shape[0], np.int8)
        root = _ColorNode()
        root.class_id = 1
        
        def _get_class_mean_cov(colors, classes, node):
            curr_node_colors = colors[np.where(classes == node.class_id)]
            node.mean = curr_node_colors.mean(axis=0)
            node.cov = np.cov(curr_node_colors.T)
            node.num_pixel = curr_node_colors.shape[0]
        
        def _get_max_eigenvalue_node(curr_node):
            queue = deque()
            max_eigen = -1
            queue.append(curr_node)
            if not (curr_node.left or curr_node.right):
                return curr_node
            while len(queue):
                node = queue.popleft()
                if node.left and node.right:
                    queue.append(node.left)
                    queue.append(node.right)
                    continue
                eigen_vals, eigen_vecs = LA.eig(node.cov)
                eigen_val = eigen_vals.max()
                if eigen_val > max_eigen:
                    max_eigen = eigen_val
                    ret = node
            return ret
        
        def _get_next_class_id(root):
            max_id = 0
            queue = deque()
            queue.append(root)
            while len(queue):
                curr_node = queue.popleft()
                if curr_node.class_id > max_id:
                    max_id = curr_node.class_id
                if curr_node.left:
                    queue.append(curr_node.left)
                if curr_node.right:
                    queue.append(curr_node.right)
            return max_id + 1
        
        def _partition_class(colors, classes, next_id, node):
            class_id = node.class_id
            left_id = next_id
            right_id = next_id + 1
            eigen_vals, eigen_vecs = LA.eig(node.cov)
            eigen_vec = eigen_vecs[eigen_vals.argmax()]
            threshold = np.dot(node.mean, eigen_vec)
            color_indices = np.where(classes == class_id)[0]
            curr_colors = colors[color_indices]
            products = np.dot(curr_colors, eigen_vec)
            left_indices = color_indices[np.where(products <= threshold)[0]]
            right_indices = color_indices[np.where(products > threshold)[0]]
            classes[left_indices] = left_id
            classes[right_indices] = right_id
            node.left = _ColorNode()
            node.left.class_id = left_id
            node.right = _ColorNode()
            node.right.class_id = right_id
        
        def _get_dominants(root):
            dominant_colors = []
            queue = deque()
            queue.append(root)
            while len(queue):
                curr_node = queue.popleft()
                if curr_node.left and curr_node.right:
                    queue.append(curr_node.left)
                    queue.append(curr_node.right)
                    continue
                color = curr_node.mean * 255
                color = np.clip(color, 0, 255)
                color = color.astype(np.uint8)
                dominant_colors.append([curr_node.num_pixel, color.tolist()])
            dominant_colors.sort(key=lambda x: x[0], reverse=True)
            return [color[1] for color in dominant_colors]
        
        _get_class_mean_cov(colors, classes, root)
        for _ in range(count - 1):
            next_node = _get_max_eigenvalue_node(root)
            next_class_id = _get_next_class_id(root)
            _partition_class(colors, classes, next_class_id, next_node)
            _get_class_mean_cov(colors, classes, next_node.left)
            _get_class_mean_cov(colors, classes, next_node.right)
        return _get_dominants(root)
    
    def _list2tuple(l):
        tlist = []
        for e in l:
            tlist.append(tuple(e))
        return tlist
    
    im_arr = np.asarray(img)
    if img.mode == "RGBA":
        im_arr = _rgba2rgb(im_arr)
    return _list2tuple(_find_dominant_colors(im_arr, num_colors))

# Импорт функций разделения цветов из внешних модулей
try:
    from color_separation import (
        kmeans_color_separation,
        dominant_color_separation, 
        threshold_color_separation,
        lab_color_separation,
        exact_color_separation,
        combine_layers,
        change_layer_color,
        get_color_from_code,
        invert_layer,
        erode_dilate_layer,
        transform_layer,
        adjust_layer_opacity,
        apply_blur_sharpen,
        apply_threshold
    )
except ImportError:
    st.error("Ошибка: Не удалось импортировать модули разделения цветов. Убедитесь, что файл color_separation.py находится в той же директории.")
    # Заглушки функций для избежания ошибок
    def dummy_function(*args, **kwargs):
        return [], []
    
    kmeans_color_separation = dummy_function
    dominant_color_separation = dummy_function
    threshold_color_separation = dummy_function
    lab_color_separation = dummy_function
    exact_color_separation = dummy_function
    combine_layers = dummy_function
    change_layer_color = dummy_function
    get_color_from_code = dummy_function
    invert_layer = dummy_function
    erode_dilate_layer = dummy_function
    transform_layer = dummy_function
    adjust_layer_opacity = dummy_function
    apply_blur_sharpen = dummy_function
    apply_threshold = dummy_function

# Импорт кодов цветов Pantone
try:
    pantone_codes = pantone.get_all_pantone_codes()
except:
    pantone_codes = {}

# Функция для преобразования изображения в png
def convert_to_png(image_array, filename):
    """Конвертирует массив изображения в формат png"""
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_array)
        ax.axis('off')
        fig.tight_layout(pad=0)
        
        # Сохраняем как png
        png_buffer = io.BytesIO()
        plt.savefig(png_buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        png_buffer.seek(0)
        return png_buffer.getvalue()
    except Exception as e:
        st.error(f"Ошибка при создании png: {e}")
        return None

# Функция для создания черно-белой маски из цветного слоя
def create_bw_mask(layer, bg_color):
    """
    Создает маску в градациях серого из цветного слоя.
    Цвета преобразуются в соответствующие оттенки серого по яркости.
    Фон становится черным (0).
    """
    # Создаем маску для определения фона
    is_background = np.all(layer == bg_color, axis=2)
    
    # Конвертируем весь слой BGR в градации серого за один раз
    gray_image = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
    
    # Создаем копию серого изображения
    gray_mask = gray_image.copy()
    
    # Фон делаем черным (0)
    gray_mask[is_background] = 0
    
    return gray_mask

# Функция для сохранения черно-белой маски в формате png
def save_bw_mask_as_png(mask, filename):
    """Сохраняет черно-белую маску в формате png"""
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mask, cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        fig.tight_layout(pad=0)
        
        # Сохраняем как png
        png_buffer = io.BytesIO()
        plt.savefig(png_buffer, format='png', bbox_inches='tight', pad_inches=0, 
                    dpi=300, facecolor='none', edgecolor='none')
        plt.close(fig)
        
        png_buffer.seek(0)
        return png_buffer.getvalue()
    except Exception as e:
        st.error(f"Ошибка при создании ЧБ маски png: {e}")
        return None

# Функция для создания маски в формате png (старая версия)
def create_mask_png(image_array, bg_color, filename):
    """Создает черно-белую маску в формате png (устаревшая функция)"""
    try:
        # Создаем маску (белый передний план, черный фон)
        mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
        is_fg = np.logical_not(np.all(image_array == bg_color, axis=2))
        mask[is_fg] = 255
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mask, cmap='gray')
        ax.axis('off')
        fig.tight_layout(pad=0)
        
        # Сохраняем как png
        png_buffer = io.BytesIO()
        plt.savefig(png_buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        png_buffer.seek(0)
        return png_buffer.getvalue()
    except Exception as e:
        st.error(f"Ошибка при создании маски png: {e}")
        return None

# Функция для преобразования слоев decompose в формат для отображения
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
            # Используем медиану цветов, где альфа > 0.5
            mask = alpha_array > 0.5
            if np.any(mask):
                # Получаем цвета пикселей с высокой прозрачностью
                masked_colors = rgb_array[mask]
                # Вычисляем медианный цвет
                median_color = np.median(masked_colors, axis=0).astype(int)
                # Конвертируем RGB в BGR для консистентности
                median_color_bgr = (median_color[2], median_color[1], median_color[0])
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
            unique_colors, counts = np.unique(rgba_array.reshape(-1, 3), axis=0, return_counts=True)
            dominant_color_idx = np.argmax(counts)
            dominant_color_rgb = unique_colors[dominant_color_idx]
            dominant_color_bgr = (dominant_color_rgb[2], dominant_color_rgb[1], dominant_color_rgb[0])
            
            # Процент покрытия (все пиксели, кроме фона)
            non_bg_mask = np.any(bgr_layer != bg_color, axis=2)
            coverage_percentage = (np.sum(non_bg_mask) / non_bg_mask.size) * 100
            
            cv_layers.append(bgr_layer)
            color_info_list.append({
                'color': dominant_color_bgr,
                'percentage': coverage_percentage
            })
    
    return cv_layers, color_info_list

# Боковая панель для управления
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Настройки</h2>", unsafe_allow_html=True)
    
    # Загрузка изображения
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        method = st.selectbox(
            "Выберите метод разделения",
            [
                "Точное извлечение цветов",  
                "K-средних кластеризация", 
                "Извлечение доминирующих цветов",
                "Цветовая пороговая обработка",
                "Цветовое пространство LAB",
                "Fast Soft Color Segmentation (Decompose)"
            ]
        )
        
        # Параметры для точного извлечения цветов
        if method == "Точное извлечение цветов":
            max_colors = st.slider("Максимальное количество цветов для извлечения", 5, 15, 10)
            st.warning("Примечание: Изображения с градиентами или шумом могут иметь много уникальных цветов. Этот метод создает один слой для каждого уникального цвета.")
        
        # Параметры для K-средних
        elif method == "K-средних кластеризация":
            num_colors = st.slider("Количество цветов для извлечения", 2, 20, 5)
            compactness = st.slider("Компактность цветов", 0.1, 10.0, 1.0, 0.1)
        
        # Параметры для доминирующих цветов
        elif method == "Извлечение доминирующих цветов":
            num_colors = st.slider("Количество цветов для извлечения", 2, 20, 5)
            min_percentage = st.slider("Минимальный процент цвета", 0.1, 10.0, 1.0, 0.1)
        
        # Параметры для пороговой обработки
        elif method == "Цветовая пороговая обработка":
            threshold_value = st.slider("Чувствительность порога", 5, 100, 25)
            blur_amount = st.slider("Степень размытия", 0, 10, 3)
        
        # Параметры для цветового пространства LAB
        elif method == "Цветовое пространство LAB":
            num_colors = st.slider("Количество цветов для извлечения", 2, 20, 5)
            delta_e = st.slider("Порог разницы цветов (Delta E)", 1, 50, 15)
        
        # Параметры для Fast Soft Color Segmentation
        elif method == "Fast Soft Color Segmentation (Decompose)":
            num_colors = st.slider("Количество цветов для извлечения", 2, 7, 7)
            st.info("Этот метод использует нейронную сеть для быстрого разделения цветов. Всегда создает 7 слоев, но вы можете настроить количество используемых цветов.")
            use_guided_filter = st.checkbox("Использовать направленный фильтр", value=True)
            normalize_alpha = st.checkbox("Нормализовать альфа-каналы", value=True)
            resize_factor = st.slider("Коэффициент изменения размера", 0.5, 2.0, 1.0, 0.1)
        
        # Глобальные параметры
        bg_color = st.color_picker("Цвет фона", "#FFFFFF")
        bg_color_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        noise_reduction = st.slider("Уменьшение шума", 0, 10, 2)
        
        # Постобработка
        st.markdown("<h3>Постобработка</h3>", unsafe_allow_html=True)
        apply_smoothing = st.checkbox("Применить сглаживание", True)
        if apply_smoothing:
            smoothing_amount = st.slider("Степень сглаживания", 1, 15, 3, 2)
        
        apply_sharpening = st.checkbox("Применить резкость", False)
        if apply_sharpening:
            sharpening_amount = st.slider("Степень резкости", 0.1, 5.0, 1.0, 0.1)

# Информационное сообщение о формате экспорта
st.info("""
**Важно:** При скачивании слоев создаются черно-белые маски (белый = область печати, черный = фон). 
Это стандартный формат для текстильной печати, где каждый цвет печатается отдельно.
""")

# Основное содержимое
if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    # Чтение изображения
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Конвертация PIL Image в формат OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.session_state.original_image_cv = img_cv
    
    with col1:
        st.markdown("<h2 class='sub-header'>Исходное изображение</h2>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        
        # Информация об изображении
        st.markdown("<h3>Информация об изображении</h3>", unsafe_allow_html=True)
        st.write(f"Размер: {image.width} x {image.height} пикселей")
        st.write(f"Формат: {image.format}")
        st.write(f"Режим: {image.mode}")
        
    with col2:
        st.markdown("<h2 class='sub-header'>Разделенные цветовые слои</h2>", unsafe_allow_html=True)
        
        # Применение выбранного метода
        with st.spinner("Разделение цветов... Пожалуйста, подождите."):
            try:
                # Обработка изображения на основе выбранного метода
                if method == "Точное извлечение цветов":
                    color_layers, color_info = exact_color_separation(
                        img_cv,
                        max_colors=max_colors,
                        bg_color=bg_color_rgb
                    )
                
                elif method == "K-средних кластеризация":
                    color_layers, color_info = kmeans_color_separation(
                        img_cv, 
                        n_colors=num_colors,
                        compactness=compactness,
                        bg_color=bg_color_rgb,
                        noise_reduction=noise_reduction,
                        apply_smoothing=apply_smoothing,
                        smoothing_amount=smoothing_amount if apply_smoothing else 0,
                        apply_sharpening=apply_sharpening,
                        sharpening_amount=sharpening_amount if apply_sharpening else 0
                    )
                
                elif method == "Извлечение доминирующих цветов":
                    color_layers, color_info = dominant_color_separation(
                        img_cv, 
                        n_colors=num_colors,
                        min_percentage=min_percentage,
                        bg_color=bg_color_rgb,
                        noise_reduction=noise_reduction,
                        apply_smoothing=apply_smoothing,
                        smoothing_amount=smoothing_amount if apply_smoothing else 0,
                        apply_sharpening=apply_sharpening,
                        sharpening_amount=sharpening_amount if apply_sharpening else 0
                    )
                
                elif method == "Цветовая пороговая обработка":
                    color_layers, color_info = threshold_color_separation(
                        img_cv, 
                        threshold=threshold_value,
                        blur_amount=blur_amount,
                        bg_color=bg_color_rgb,
                        noise_reduction=noise_reduction,
                        apply_smoothing=apply_smoothing,
                        smoothing_amount=smoothing_amount if apply_smoothing else 0,
                        apply_sharpening=apply_sharpening,
                        sharpening_amount=sharpening_amount if apply_sharpening else 0
                    )
                
                elif method == "Цветовое пространство LAB":
                    color_layers, color_info = lab_color_separation(
                        img_cv, 
                        n_colors=num_colors,
                        delta_e=delta_e,
                        bg_color=bg_color_rgb,
                        noise_reduction=noise_reduction,
                        apply_smoothing=apply_smoothing,
                        smoothing_amount=smoothing_amount if apply_smoothing else 0,
                        apply_sharpening=apply_sharpening,
                        sharpening_amount=sharpening_amount if apply_sharpening else 0
                    )
                
                elif method == "Fast Soft Color Segmentation (Decompose)":
                    # Используем метод decompose
                    st.info("Используется метод Fast Soft Color Segmentation (нейронная сеть)")
                    
                    # Получаем доминирующие цвета для палитры
                    palette_colors = get_dominant_colors(image, num_colors)
                    
                    # Вызываем функцию decompose
                    decompose_layers = decompose_fast_soft_color(
                        image,
                        palette=palette_colors,
                        guided_filter=use_guided_filter,
                        normalize_alpha=normalize_alpha,
                        resize_scale_factor=resize_factor
                    )
                    
                    if decompose_layers:
                        # Преобразуем слои decompose в формат для отображения
                        color_layers, color_info = decompose_layers_to_cv_format(
                            decompose_layers, 
                            bg_color_rgb
                        )
                    else:
                        st.error("Не удалось выполнить разделение с помощью метода Decompose. Пожалуйста, проверьте наличие файла модели.")
                        color_layers, color_info = [], []
                
                # Сохраняем результаты в session state
                st.session_state.color_layers = color_layers
                st.session_state.color_info = color_info
                
            except Exception as e:
                st.error(f"Ошибка при разделении цветов: {e}")
                st.session_state.color_layers = []
                st.session_state.color_info = []
        
        # Используем данные из session state
        color_layers = st.session_state.color_layers
        color_info = st.session_state.color_info
        
        # Показать извлеченные слои
        if color_layers and color_info:
            for i, (layer, info) in enumerate(zip(color_layers, color_info)):
                col_left, col_right = st.columns([3, 1])
                
                with col_left:
                    # Конвертация слоя из BGR в RGB для отображения
                    layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                    st.image(layer_rgb, caption=f"Слой {i+1}", use_column_width=True)
                    
                    # Создаем черно-белую маску для скачивания
                    bw_mask = create_bw_mask(layer, bg_color_rgb)
                    png_data = save_bw_mask_as_png(bw_mask, f"mask_{i+1}")
                    
                    if png_data:
                        hex_color = "{:02x}{:02x}{:02x}".format(
                            info['color'][2], info['color'][1], info['color'][0]  # BGR в RGB
                        )
                        
                        st.download_button(
                            label=f"Скачать ЧБ маску слоя {i+1} (png)",
                            data=png_data,
                            file_name=f"mask_{i+1}_{hex_color}.png",
                            mime="application/postscript",
                            key=f"download_layer_mask_{i}"
                        )
                
                with col_right:
                    hex_color = "#{:02x}{:02x}{:02x}".format(
                        info['color'][2], info['color'][1], info['color'][0]  # BGR в RGB
                    )
                    st.markdown(
                        f"<div><span class='color-chip' style='background-color: {hex_color}'></span> {hex_color}</div>",
                        unsafe_allow_html=True
                    )
                    st.write(f"RGB: {info['color'][::-1]}")  # BGR в RGB
                    st.write(f"Покрытие: {info['percentage']:.1f}%")
        else:
            st.warning("Не удалось извлечь цветовые слои. Попробуйте другие настройки.")

# Создание комбинированного предпросмотра
if uploaded_file is not None and st.session_state.color_layers and st.session_state.color_info:
    color_layers = st.session_state.color_layers
    color_info = st.session_state.color_info
    
    if len(color_layers) > 0:
        st.markdown("""
        <div style='background-color: #f0f8ff; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
            <h3>Комбинированный предпросмотр</h3>
            <p>Управляйте визуальным порядком слоев в вашем предпросмотре. Используйте настройки порядка и видимости слоев ниже, чтобы:</p>
            <ul>
                <li>Изменить порядок наложения слоев (слои с более высокими номерами позиций отображаются сверху)</li>
                <li>Включить/выключить видимость слоев для предпросмотра различных комбинаций</li>
                <li>Сохранить текущее расположение слоев для дальнейшего редактирования</li>
            </ul>
            <p>Все загрузки будут учитывать ваши настройки порядка и видимости слоев.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Добавление опции для изменения порядка слоев
        with st.expander("Настройки порядка и видимости слоев", expanded=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.write("Управляйте порядком наложения и видимостью ваших слоев:")
            
            with col2:
                # Добавить кнопку для сброса видимости всех слоев
                if st.button("Показать все слои", key="show_all_layers_main"):
                    st.session_state.layer_visibility = [True] * len(color_layers)
                    st.rerun()
            
            # Инициализация состояния сессии для порядка и видимости
            if 'layer_order' not in st.session_state or len(st.session_state.layer_order) != len(color_layers):
                st.session_state.layer_order = list(range(len(color_layers)))
            if 'layer_visibility' not in st.session_state or len(st.session_state.layer_visibility) != len(color_layers):
                st.session_state.layer_visibility = [True] * len(color_layers)
            
            # Создание колонок для каждого слоя для изменения порядка и переключения видимости
            for i in range(len(color_layers)):
                col1, col2, col3 = st.columns([2, 1, 3])
                
                with col1:
                    # Уникальный ключ для каждого number_input
                    order_value = st.number_input(
                        f"Позиция слоя {i+1}",
                        min_value=1,
                        max_value=len(color_layers),
                        value=st.session_state.layer_order[i] + 1,
                        key=f"layer_order_number_{i}"
                    )
                    # Обновляем состояние только если значение изменилось
                    if st.session_state.layer_order[i] != order_value - 1:
                        st.session_state.layer_order[i] = order_value - 1
                
                with col2:
                    # Уникальный ключ для каждого checkbox
                    visibility_state = st.checkbox(
                        "Видимый",
                        value=st.session_state.layer_visibility[i],
                        key=f"layer_visibility_checkbox_{i}"
                    )
                    if st.session_state.layer_visibility[i] != visibility_state:
                        st.session_state.layer_visibility[i] = visibility_state
                
                with col3:
                    # Отображение образца цвета для этого слоя
                    hex_color = "#{:02x}{:02x}{:02x}".format(
                        color_info[i]['color'][2], color_info[i]['color'][1], color_info[i]['color'][0]  # BGR в RGB
                    )
                    st.markdown(
                        f"<div style='display: flex; align-items: center; gap: 10px; padding: 5px;'>"
                        f"<div style='background-color: {hex_color}; width: 30px; height: 30px; border: 1px solid #000; border-radius: 4px;'></div>"
                        f"<div>"
                        f"<div style='font-weight: bold;'>Слой {i+1}</div>"
                        f"<div style='font-size: 0.8em; color: #666;'>{hex_color} - {color_info[i]['percentage']:.1f}%</div>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                # Разделитель между слоями
                if i < len(color_layers) - 1:
                    st.markdown("---")
        
        # Создание комбинированного изображения на основе пользовательского порядка
        combined = np.zeros_like(st.session_state.original_image_cv, dtype=np.uint8)
        
        # Создаем список слоев в правильном порядке (от нижнего к верхнему)
        sorted_indices = sorted(range(len(st.session_state.layer_order)), 
                               key=lambda x: st.session_state.layer_order[x])
        
        # Применение слоев в правильном порядке (от нижнего к верхнему)
        for idx in sorted_indices:
            if st.session_state.layer_visibility[idx]:
                layer = color_layers[idx]
                
                # Создаем маску для текущего слоя (где есть цвет, отличный от фона)
                mask = np.any(layer != bg_color_rgb, axis=2)
                
                # Для областей, где маска True, берем пиксели из текущего слоя
                # Для областей, где маска False, оставляем пиксели из combined
                combined[mask] = layer[mask]
        
        # Конвертация комбинированного из BGR в RGB для отображения
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        
        # Подсчет видимых слоев
        visible_layers = sum(st.session_state.layer_visibility)
        total_layers = len(color_layers)
        
        # Создание подписи, показывающей статус порядка слоев
        if visible_layers == total_layers:
            caption = f"Комбинированный предпросмотр всех {total_layers} слоев с пользовательским порядком"
        else:
            caption = f"Комбинированный предпросмотр {visible_layers}/{total_layers} видимых слоев с пользовательским порядком"
            
        st.image(combined_rgb, caption=caption, use_column_width=True)
        
        # Создаем черно-белую маску комбинированного изображения для скачивания
        combined_bw_mask = np.zeros((combined_rgb.shape[0], combined_rgb.shape[1]), dtype=np.uint8)
        
        # Создаем маску для каждого видимого слоя и комбинируем их
        for i, layer in enumerate(color_layers):
            if st.session_state.layer_visibility[i]:
                layer_mask = create_bw_mask(layer, bg_color_rgb)
                combined_bw_mask = cv2.bitwise_or(combined_bw_mask, layer_mask)
        
        # Кнопка скачивания для комбинированной черно-белой маски в формате png
        png_data = save_bw_mask_as_png(combined_bw_mask, "combined_mask")
        
        col1, col2 = st.columns(2)
        with col1:
            if png_data:
                st.download_button(
                    label="Скачать комбинированную ЧБ маску (png)",
                    data=png_data,
                    file_name="combined_mask.png",
                    mime="application/postscript",
                    key="download_combined_mask_main"
                )
        
        with col2:
            # Создание кнопки для сохранения порядка слоев
            if st.button("Сохранить текущий порядок слоев", key="save_layer_order_main"):
                if 'custom_layers' not in st.session_state:
                    st.session_state.custom_layers = []
                
                # Добавление упорядоченного комбинированного изображения в пользовательские слои
                st.session_state.custom_layers.append({
                    'layer': combined,
                    'name': f"Комбинированный с пользовательским порядком ({visible_layers}/{total_layers} слоев)"
                })
                
                st.success("Текущий порядок слоев сохранен в Галерею обработанных слоев!")
        
        # Опции скачивания
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h3>Опции скачивания</h3>
            <p>Выберите из различных форматов скачивания для вашего рабочего процесса текстильной печати.</p>
        </div>
        """, unsafe_allow_html=True)
        
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            if st.button("Подготовить пакет всех черно-белых масок", key="prepare_all_bw_masks"):
                with st.spinner("Подготовка черно-белых масок для скачивания..."):
                    # Создание временной директории
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        # Определение, есть ли у нас упорядоченный список слоев
                        has_ordered_layers = 'layer_order' in st.session_state and len(st.session_state.layer_order) == len(color_layers)
                        has_visibility = 'layer_visibility' in st.session_state and len(st.session_state.layer_visibility) == len(color_layers)
                        
                        # Сохранение каждого слоя как черно-белой маски
                        mask_files = []
                        for i, layer in enumerate(color_layers):
                            # Получение позиции слоя в стеке (если упорядочивание активно)
                            position = i
                            if has_ordered_layers:
                                # Нахождение позиции этого слоя в упорядоченном списке
                                position = st.session_state.layer_order[i]
                                
                            # Пропуск слоев, которые установлены как невидимые
                            if has_visibility and not st.session_state.layer_visibility[i]:
                                continue
                            
                            # Создание черно-белой маски
                            bw_mask = create_bw_mask(layer, bg_color_rgb)
                            
                            # Сохранение маски в формате png
                            hex_color = "{:02x}{:02x}{:02x}".format(
                                color_info[i]['color'][2], color_info[i]['color'][1], color_info[i]['color'][0]
                            )
                            # Включение позиции в имя файла
                            mask_filename = f"position{position+1:02d}_mask_{i+1}_{hex_color}.png"
                            mask_path = os.path.join(tmpdirname, mask_filename)
                            
                            # Сохранение как png
                            png_data = save_bw_mask_as_png(bw_mask, mask_filename)
                            if png_data:
                                with open(mask_path, 'wb') as f:
                                    f.write(png_data)
                                mask_files.append(mask_path)
                        
                        # Также сохраняем комбинированный слой как черно-белую маску
                        if len(mask_files) > 0:
                            # Создание комбинированной черно-белой маски
                            combined_mask = np.zeros((color_layers[0].shape[0], color_layers[0].shape[1]), dtype=np.uint8)
                            
                            # Наложение всех видимых масок
                            for i, layer in enumerate(color_layers):
                                if has_visibility and not st.session_state.layer_visibility[i]:
                                    continue
                                
                                # Создаем маску для текущего слоя
                                layer_mask = create_bw_mask(layer, bg_color_rgb)
                                
                                # Добавляем к комбинированной маске
                                combined_mask = cv2.bitwise_or(combined_mask, layer_mask)
                            
                            # Сохраняем комбинированную маску
                            combined_mask_path = os.path.join(tmpdirname, "combined_mask.png")
                            png_data = save_bw_mask_as_png(combined_mask, "combined_mask")
                            if png_data:
                                with open(combined_mask_path, 'wb') as f:
                                    f.write(png_data)
                        
                        # Создание текстового файла README с объяснением
                        readme_content = """# ColorSep Экспортированные маски для текстильной печати

Этот пакет содержит черно-белые маски для каждого цветового слоя.

## Формат файлов
- Все файлы - это черно-белые PNG маски
- Белый цвет (255) = область печати
- Черный цвет (0) = фон (не печатается)
- Каждый файл соответствует одному цветовому слою

## Соглашение об именах файлов
- Файлы названы по шаблону: position{XX}_mask_{Y}_{color}.png
- Position: Позиция в порядке наложения (01 - нижний слой, более высокие номера сверху)
- Mask: Черно-белая маска слоя
- Y: Исходный номер слоя из извлечения
- Color: HEX-код оригинального цвета слоя (только для справки)

## Содержимое
- Отдельные маски для каждого цветового слоя
- Combined_mask.png: Комбинированная маска всех видимых слоев
"""
                        
                        # Добавление информации о каждом слое в README
                        readme_content += "\n\n## Детали слоев\n"
                        for i, layer in enumerate(color_layers):
                            position = i
                            if 'layer_order' in st.session_state and len(st.session_state.layer_order) == len(color_layers):
                                position = st.session_state.layer_order[i]
                            
                            hex_color = "{:02x}{:02x}{:02x}".format(
                                color_info[i]['color'][2], color_info[i]['color'][1], color_info[i]['color'][0]
                            )
                            
                            # Проверка видимости
                            is_visible = True
                            if 'layer_visibility' in st.session_state and len(st.session_state.layer_visibility) == len(color_layers):
                                is_visible = st.session_state.layer_visibility[i]
                            
                            readme_content += f"- Слой {i+1}: Позиция {position+1}, Цвет #{hex_color}, Видим: {'Да' if is_visible else 'Нет'}, Покрытие {color_info[i]['percentage']:.1f}%\n"
                        
                        # Сохранение файла README
                        readme_path = os.path.join(tmpdirname, "README.txt")
                        with open(readme_path, 'w', encoding='utf-8') as f:
                            f.write(readme_content)
                        
                        # Создание zip-файла
                        zip_path = os.path.join(tmpdirname, "bw_masks.zip")
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for file in mask_files:
                                zipf.write(file, os.path.basename(file))
                            if os.path.exists(combined_mask_path):
                                zipf.write(combined_mask_path, os.path.basename(combined_mask_path))
                            zipf.write(readme_path, os.path.basename(readme_path))
                        
                        # Чтение zip  файла
                        with open(zip_path, "rb") as f:
                            zip_data = f.read()
                        
                        # Предоставление ссылки для скачивания
                        st.download_button(
                            label="Скачать ЧБ маски всех слоев (ZIP)",
                            data=zip_data,
                            file_name="bw_color_masks.zip",
                            mime="application/zip",
                            key="download_all_bw_masks_zip"
                        )
                        
        with download_col2:
            if st.button("Сохранить как цветные png", key="save_color_pngs"):
                with st.spinner("Подготовка цветных файлов для скачивания..."):
                    # Создание временной директории
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        # Определение, есть ли у нас упорядоченный список слоев
                        has_ordered_layers = 'layer_order' in st.session_state and len(st.session_state.layer_order) == len(color_layers)
                        has_visibility = 'layer_visibility' in st.session_state and len(st.session_state.layer_visibility) == len(color_layers)
                        
                        # Сохранение каждого слоя как цветного png
                        color_files = []
                        for i, layer in enumerate(color_layers):
                            # Получение позиции слоя в стеке (если упорядочивание активно)
                            position = i
                            if has_ordered_layers:
                                # Нахождение позиции этого слоя в упорядоченном списке
                                position = st.session_state.layer_order[i]
                                
                            # Пропуск слоев, которые установлены как невидимые
                            if has_visibility and not st.session_state.layer_visibility[i]:
                                continue
                            
                            # Конвертация BGR в RGB перед сохранением
                            layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                            
                            # Сохранение как png
                            hex_color = "{:02x}{:02x}{:02x}".format(
                                color_info[i]['color'][2], color_info[i]['color'][1], color_info[i]['color'][0]
                            )
                            # Включение позиции в имя файла
                            color_filename = f"position{position+1:02d}_layer_{i+1}_{hex_color}.png"
                            color_path = os.path.join(tmpdirname, color_filename)
                            
                            png_data = convert_to_png(layer_rgb, color_filename)
                            if png_data:
                                with open(color_path, 'wb') as f:
                                    f.write(png_data)
                                color_files.append(color_path)
                        
                        # Создание zip-файла
                        zip_path = os.path.join(tmpdirname, "color_layers.zip")
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for file in color_files:
                                zipf.write(file, os.path.basename(file))
                        
                        # Чтение zip-файла
                        with open(zip_path, "rb") as f:
                            zip_data = f.read()
                        
                        # Предоставление ссылки для скачивания
                        st.download_button(
                            label="Скачать цветные слои (ZIP)",
                            data=zip_data,
                            file_name="color_layers.zip",
                            mime="application/zip",
                            key="download_color_layers_zip"
                        )
        
        # Инструменты манипуляции со слоями
        st.markdown("""
        <div style='background-color: #f2f8f3; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
            <h3>Инструменты манипуляции со слоями</h3>
            <p>Объединяйте слои или изменяйте их цвета для достижения идеального разделения для вашего проекта текстильной печати.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Объединить слои"):
            if len(color_layers) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    layer1_idx = st.selectbox(
                        "Выберите первый слой",
                        range(len(color_layers)),
                        format_func=lambda i: f"Слой {i+1} - {color_info[i]['percentage']:.1f}%",
                        key="combine_layer1"
                    )
                with col2:
                    layer2_idx = st.selectbox(
                        "Выберите второй слой",
                        range(len(color_layers)),
                        format_func=lambda i: f"Слой {i+1} - {color_info[i]['percentage']:.1f}%",
                        index=min(1, len(color_layers)-1),  # По умолчанию второй слой
                        key="combine_layer2"
                    )
                
                use_custom_color = st.checkbox("Использовать пользовательский цвет для объединенного слоя", key="use_custom_color")
                custom_color = None
                
                if use_custom_color:
                    color_input_method = st.radio(
                        "Метод ввода цвета",
                        ["Палитра цветов", "Значение RGB", "HEX код", "Pantone TPX/TPG"],
                        horizontal=True,
                        key="color_input_method"
                    )
                    
                    if color_input_method == "Палитра цветов":
                        custom_color_hex = st.color_picker("Выберите цвет", "#FF0000", key="color_picker")
                        custom_color = get_color_from_code(custom_color_hex)
                    
                    elif color_input_method == "Значение RGB":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            r_val = st.number_input("R", 0, 255, 255, key="r_val")
                        with col2:
                            g_val = st.number_input("G", 0, 255, 0, key="g_val")
                        with col3:
                            b_val = st.number_input("B", 0, 255, 0, key="b_val")
                        custom_color = (b_val, g_val, r_val)  # BGR формат для OpenCV
                    
                    elif color_input_method == "HEX код":
                        hex_val = st.text_input("HEX код (например, #FF0000)", "#FF0000", key="hex_val")
                        custom_color = get_color_from_code(hex_val)
                    
                    elif color_input_method == "Pantone TPX/TPG":
                        pantone_code_type = st.selectbox(
                            "Выберите тип кода Pantone",
                            ["TPX", "TPG"],
                            key="pantone_code_type_combine"
                        )
                        
                        # Здесь должна быть логика для работы с Pantone
                        st.info("Функциональность Pantone требует дополнительной настройки")
                
                if st.button("Объединить слои", key="combine_layers_btn"):
                    with st.spinner("Объединение слоев..."):
                        try:
                            # Получение выбранных слоев
                            layer1 = color_layers[layer1_idx]
                            layer2 = color_layers[layer2_idx]
                            
                            # Объединение слоев
                            combined_layer = combine_layers(layer1, layer2, custom_color, bg_color_rgb)
                            
                            # Расчет процента объединенного слоя
                            h, w = combined_layer.shape[:2]
                            mask = np.zeros((h, w), dtype=np.uint8)
                            is_fg = np.logical_not(np.all(combined_layer == bg_color_rgb, axis=2))
                            mask[is_fg] = 255
                            percentage = (np.sum(mask) / 255 / (h * w)) * 100
                            
                            # Установка цвета для объединенного слоя
                            if custom_color:
                                new_color = custom_color
                            else:
                                # Использование цвета из layer1, если нет пользовательского цвета
                                new_color = color_info[layer1_idx]['color']
                            
                            # Создаем копии для избежания изменения оригинальных данных
                            updated_color_layers = color_layers.copy()
                            updated_color_info = color_info.copy()
                            
                            # Удаление исходных слоев
                            replaced_indices = sorted([layer1_idx, layer2_idx], reverse=True)
                            for idx in replaced_indices:
                                updated_color_layers.pop(idx)
                                updated_color_info.pop(idx)
                            
                            # Добавление объединенного слоя
                            updated_color_layers.append(combined_layer)
                            updated_color_info.append({
                                'color': new_color,
                                'percentage': percentage
                            })
                            
                            # Обновляем session state
                            st.session_state.color_layers = updated_color_layers
                            st.session_state.color_info = updated_color_info
                            
                            # Отображение результата
                            result_rgb = cv2.cvtColor(combined_layer, cv2.COLOR_BGR2RGB)
                            st.image(result_rgb, caption="Объединенный слой", use_column_width=True)
                            
                            # Создаем черно-белую маску для скачивания
                            bw_mask = create_bw_mask(combined_layer, bg_color_rgb)
                            png_data = save_bw_mask_as_png(bw_mask, f"combined_mask_{layer1_idx+1}_{layer2_idx+1}")
                            
                            if png_data:
                                hex_color = "{:02x}{:02x}{:02x}".format(
                                    new_color[2], new_color[1], new_color[0]  # BGR в RGB
                                )
                                
                                st.download_button(
                                    label=f"Скачать ЧБ маску объединенного слоя (png)",
                                    data=png_data,
                                    file_name=f"combined_mask_{layer1_idx+1}_{layer2_idx+1}.png",
                                    mime="application/postscript",
                                    key=f"download_combined_mask_{layer1_idx}_{layer2_idx}"
                                )
                            
                            # Сохранение этого нового слоя в состоянии сессии
                            if 'custom_layers' not in st.session_state:
                                st.session_state.custom_layers = []
                            
                            st.session_state.custom_layers.append({
                                'layer': combined_layer,
                                'name': f"Объединенный {layer1_idx+1} & {layer2_idx+1}"
                            })
                            
                            st.success(f"Слои {layer1_idx+1} и {layer2_idx+1} успешно объединены!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Ошибка при объединении слоев: {e}")
            else:
                st.warning("Вам нужно как минимум 2 слоя для использования этой функции")
        
        with st.expander("Изменить цвет слоя"):
            if len(color_layers) > 0:
                # Выбор слоя для изменения
                layer_idx = st.selectbox(
                    "Выберите слой для перекрашивания",
                    range(len(color_layers)),
                    format_func=lambda i: f"Слой {i+1} - {color_info[i]['percentage']:.1f}%",
                    key="recolor_layer_select"
                )
                
                # Метод ввода цвета
                color_input_method = st.radio(
                    "Метод ввода цвета",
                    ["Палитра цветов", "Значение RGB", "HEX код", "Pantone TPX/TPG"],
                    horizontal=True,
                    key="recolor_method"
                )
                
                new_color = None
                
                if color_input_method == "Палитра цветов":
                    # Получение текущего цвета в HEX
                    current_color = color_info[layer_idx]['color']
                    current_hex = "#{:02x}{:02x}{:02x}".format(
                        current_color[2], current_color[1], current_color[0]
                    )
                    new_color_hex = st.color_picker("Выберите новый цвет", current_hex, key="recolor_picker")
                    new_color = get_color_from_code(new_color_hex)
                
                elif color_input_method == "Значение RGB":
                    col1, col2, col3 = st.columns(3)
                    # Получение текущего цвета
                    current_color = color_info[layer_idx]['color']
                    
                    with col1:
                        r_val = st.number_input("R", 0, 255, current_color[2], key="recolor_r")
                    with col2:
                        g_val = st.number_input("G", 0, 255, current_color[1], key="recolor_g")
                    with col3:
                        b_val = st.number_input("B", 0, 255, current_color[0], key="recolor_b")
                    new_color = (b_val, g_val, r_val)  # BGR формат для OpenCV
                
                elif color_input_method == "HEX код":
                    current_color = color_info[layer_idx]['color']
                    current_hex = "#{:02x}{:02x}{:02x}".format(
                        current_color[2], current_color[1], current_color[0]
                    )
                    hex_val = st.text_input("HEX код (например, #FF0000)", current_hex, key="recolor_hex")
                    new_color = get_color_from_code(hex_val)
                
                elif color_input_method == "Pantone TPX/TPG":
                    st.info("Функциональность Pantone требует дополнительной настройки")
                    new_color = color_info[layer_idx]['color']  # Оставляем текущий цвет
                
                # Предпросмотр цвета
                if new_color is not None:
                    st.markdown(
                        f"<div><span class='color-chip' style='background-color: #{new_color[2]:02x}{new_color[1]:02x}{new_color[0]:02x}; width: 50px; height: 30px;'></span> Выбранный цвет: RGB({new_color[2]}, {new_color[1]}, {new_color[0]})</div>",
                        unsafe_allow_html=True
                    )
                
                # Кнопка применения
                if st.button("Применить новый цвет", key="apply_recolor"):
                    with st.spinner("Изменение цвета слоя..."):
                        try:
                            # Получение выбранного слоя
                            layer = color_layers[layer_idx]
                            
                            # Изменение цвета
                            recolored_layer = change_layer_color(layer, new_color, bg_color_rgb)
                            
                            # Обновляем данные в session state
                            updated_color_layers = color_layers.copy()
                            updated_color_info = color_info.copy()
                            
                            updated_color_layers[layer_idx] = recolored_layer
                            updated_color_info[layer_idx]['color'] = new_color
                            
                            st.session_state.color_layers = updated_color_layers
                            st.session_state.color_info = updated_color_info
                            
                            # Отображение результатов
                            recolored_rgb = cv2.cvtColor(recolored_layer, cv2.COLOR_BGR2RGB) 
                            st.image(recolored_rgb, caption=f"Слой {layer_idx+1} с новым цветом", use_column_width=True)
                            
                            # Создаем черно-белую маску для скачивания
                            bw_mask = create_bw_mask(recolored_layer, bg_color_rgb)
                            png_data = save_bw_mask_as_png(bw_mask, f"recolored_mask_{layer_idx+1}")
                            
                            if png_data:
                                hex_color = "{:02x}{:02x}{:02x}".format(
                                    new_color[2], new_color[1], new_color[0]  # BGR в RGB
                                )
                                
                                st.download_button(
                                    label=f"Скачать ЧБ маску перекрашенного слоя (png)",
                                    data=png_data,
                                    file_name=f"recolored_mask_{layer_idx+1}_{hex_color}.png",
                                    mime="application/postscript",
                                    key=f"download_recolored_mask_{layer_idx}"
                                )
                            
                            # Сохранение этого перекрашенного слоя в состоянии сессии
                            if 'custom_layers' not in st.session_state:
                                st.session_state.custom_layers = []
                            
                            hex_color = "{:02x}{:02x}{:02x}".format(
                                new_color[2], new_color[1], new_color[0]  # BGR в RGB
                            )
                            
                            st.session_state.custom_layers.append({
                                'layer': recolored_layer,
                                'name': f"Слой {layer_idx+1} перекрашен в #{hex_color}"
                            })
                            
                            # Сообщение об успехе
                            st.success(f"Слой {layer_idx+1} был перекрашен!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Ошибка при изменении цвета: {e}")
            else:
                st.warning("Нет доступных слоев для перекрашивания")

        # Галерея обработанных слоев
        if 'custom_layers' in st.session_state and len(st.session_state.custom_layers) > 0:
            st.markdown("""
            <div style='background-color: #f0f7ff; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
                <h3>Галерея обработанных слоев</h3>
                <p>Просматривайте и скачивайте ваши пользовательские объединенные и перекрашенные слои из этой сессии.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Создание списка всех имен слоев для селектора
            layer_names = [layer_info['name'] for layer_info in st.session_state.custom_layers]
            
            # Выбор слоя для просмотра
            selected_layer_name = st.selectbox(
                "Выберите обработанный слой для просмотра",
                layer_names,
                key="custom_layer_selector"
            )
            
            # Нахождение выбранного слоя
            selected_idx = layer_names.index(selected_layer_name)
            selected_layer = st.session_state.custom_layers[selected_idx]['layer']
            
            # Отображение выбранного слоя
            selected_layer_rgb = cv2.cvtColor(selected_layer, cv2.COLOR_BGR2RGB)
            st.image(selected_layer_rgb, caption=selected_layer_name, use_column_width=True)
            
            # Создание черно-белой маски для скачивания
            bw_mask = create_bw_mask(selected_layer, bg_color_rgb)
            png_data = save_bw_mask_as_png(bw_mask, f"custom_mask_{selected_layer_name}")
            
            if png_data:
                st.download_button(
                    label="Скачать ЧБ маску этого слоя (png)",
                    data=png_data,
                    file_name=f"custom_mask_{selected_layer_name.replace(' ', '_')}.png",
                    mime="application/postscript",
                    key=f"download_custom_mask_{selected_idx}"
                )
            
            # Опция создания цветного png
            if st.button("Создать цветной png", key="create_color_png_custom"):
                # Создание цветного png
                png_data = convert_to_png(selected_layer_rgb, selected_layer_name)
                
                if png_data:
                    st.download_button(
                        label="Скачать цветной слой (png)",
                        data=png_data,
                        file_name=f"color_{selected_layer_name.replace(' ', '_')}.png",
                        mime="application/postscript",
                        key=f"download_custom_color_{selected_idx}"
                    )

else:
    # Отображение примера использования, когда изображение не загружено
    st.markdown("<h2 class='sub-header'>Как использовать ColorSep</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    1. Загрузите изображение с помощью загрузчика файлов на боковой панели
    2. Выберите метод разделения цветов:
       - **Точное извлечение цветов**: Создает один слой для каждого уникального цвета, сохраняя все детали
       - **K-средних кластеризация**: Сегментирует изображение на отдельные цветовые кластеры
       - **Извлечение доминирующих цветов**: Извлекает наиболее распространенные цвета
       - **Цветовая пороговая обработка**: Использует пороги для разделения цветов
       - **Цветовое пространство LAB**: Использует перцепционные различия цветов для более точного разделения
       - **Fast Soft Color Segmentation (Decompose)**: Использует нейронную сеть для быстрого разделения цветов на 7 слоев с прозрачностью
    3. Настройте параметры для точной настройки разделения
    4. Просмотрите каждый цветовой слой и комбинированный предпросмотр
    5. Скачайте черно-белые маски слоев для текстильной печати
    
    **Формат экспорта:** Все скачивания создают черно-белые PNG маски:
    - Белый цвет (255) = область печати
    - Черный цвет (0) = фон (не печатается)
    - Каждый цветовой слой экспортируется как отдельная маска
    
    Этот инструмент идеален для текстильной печати, где каждый цвет нужно печатать отдельно.
    
    ### Продвинутые функции:
    - **Объединение слоев**: Объедините два цветовых слоя в один слой
    - **Изменение цветов слоев**: Измените цвет любого слоя с использованием RGB, HEX или кодов цветов Pantone
    - **Экспорт в PNG**: Все скачивания в формате PNG, идеальном для полиграфии и текстильной печати
    - **Fast Soft Color Segmentation**: Новый метод на основе нейронной сети для быстрого и качественного разделения цветов
    """)
    
    st.info("⬅️ Используйте боковую панель, чтобы загрузить ваше изображение и начать работу!")

# Информация о методе Decompose
st.markdown("""
---
### О методе Fast Soft Color Segmentation (Decompose)

**Fast Soft Color Segmentation** - это метод на основе нейронной сети, который:
1. Быстро разлагает изображение на 7 цветовых слоев с прозрачностью (альфа-каналы)
2. Использует предварительно обученную модель для мгновенного разделения
3. Создает мягкие границы между цветовыми областями (soft segmentation)
4. Идеально подходит для изображений с градиентами и мягкими переходами цветов

**Особенности:**
- Всегда создает 7 слоев (можно настроить количество используемых цветов)
- Каждый слой имеет альфа-канал для плавных переходов
- Быстрая обработка даже для больших изображений
- Сохраняет мелкие детали и текстуры

**Использование в текстильной печати:**
- Позволяет создавать сложные многослойные дизайны
- Идеально для изображений с плавными цветовыми переходами
- Сохраняет художественные эффекты и тонкие детали
""")

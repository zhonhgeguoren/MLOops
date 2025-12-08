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
import sys
from pathlib import Path
import warnings
import math
from scipy.spatial.distance import mahalanobis
from scipy import linalg
import time
warnings.filterwarnings('ignore')

# ==================== КОНСТАНТЫ ИЗ C++ КОДА ====================

class Constants:
    sigma = 0.1
    tau = 0.1
    tol = 1e-4
    isMin_max_iter = 2
    cg_max_iter = 100
    gamma = 0.25
    beta = 2.0

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def read_mat_from_txt(filename, rows, cols):
    """Читает матрицу из текстового файла"""
    with open(filename, 'r') as f:
        nums = []
        for line in f:
            values = line.strip().split()
            for val in values:
                if val:
                    nums.append(float(val))
    
    mat = np.array(nums).reshape(rows, cols)
    return mat

def write_mat_to_file(mat, filename):
    """Записывает матрицу в файл"""
    with open(filename, 'w') as fout:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                fout.write(f"{mat[i, j]}\t")
            fout.write("\n")

# ==================== КЛАССЫ ИЗ COLORMODEL.H ====================

def compute_mahal_proj(p, normal, cov, color):
    """
    Вычисляет 2D расстояние Махаланобиса при проекции на плоскость
    Eq 16 of "Unmixing-Based Soft Color Segmentation for Image Manipulation"
    """
    # Нормализуем нормаль
    nn = normal / np.linalg.norm(normal)
    z = np.array([0, 0, 1])
    
    # Создаем матрицу поворота для выравнивания нормали с осью z
    v = np.cross(nn, z)
    s = np.linalg.norm(v)
    c = np.dot(nn, z)
    
    if s == 0:
        # Если векторы коллинеарны
        if c > 0:
            U = np.eye(3)
        else:
            U = -np.eye(3)
    else:
        v_mat = (z - c * nn) / np.linalg.norm(z - c * nn)
        w_mat = np.cross(z, nn)
        
        # Создаем матрицу G
        G = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])
        
        # Создаем матрицу F
        F = np.column_stack((nn.reshape(3, 1), 
                             v_mat.reshape(3, 1), 
                             w_mat.reshape(3, 1)))
        F_inv = np.linalg.inv(F)
        
        # Матрица поворота U
        U = F_inv @ G @ F
    
    # Транспонируем U
    U_trans = U.T
    
    # Преобразуем цвет
    c_rot = U @ (color - p)
    c_proj = c_rot[:2]  # проекция на плоскость xy
    
    # Преобразуем ковариацию
    cov_rot = U @ cov @ U_trans
    
    # Проекция ковариации на плоскость xy
    cov_proj = cov_rot[:2, :2]
    
    # Вычисляем расстояние Махаланобиса
    try:
        cov_inv = np.linalg.inv(cov_proj)
        diff = c_proj - np.zeros(2)
        mahal = diff.T @ cov_inv @ diff
    except np.linalg.LinAlgError:
        # Если матрица вырожденная
        mahal = 1e6
    
    return mahal

def projected_cu_energy_alt(m1, m2, c1, c2, color):
    """Projected Color Unmixing Energy - Eq. 16"""
    n = (m1 - m2) / np.linalg.norm(m1 - m2)
    dist = np.linalg.norm(m1 - m2)
    
    # Проверяем, лежит ли цвет между двумя плоскостями
    d1 = np.abs(np.dot(color - m1, n))
    d2 = np.abs(np.dot(color - m2, n))
    
    if d1 > dist or d2 > dist:
        return 100000.0  # очень высокая стоимость
    
    # Вычисляем F
    u1 = color - ((np.dot(color - m1, n)) / np.dot(n, n)) * n
    u2 = color - ((np.dot(color - m2, n)) / np.dot(n, n)) * n
    
    if np.linalg.norm(u1 - u2) == 0:
        alpha_1 = 0.5
    else:
        alpha_1 = np.linalg.norm(color - u2) / np.linalg.norm(u1 - u2)
    
    alpha_2 = 1 - alpha_1
    
    # Вычисляем расстояния Махаланобиса
    mahal1 = compute_mahal_proj(m1, m2 - m1, c1, color)
    mahal2 = compute_mahal_proj(m2, m1 - m2, c2, color)
    
    cost = alpha_1 * mahal1 + alpha_2 * mahal2
    return cost

def representation_score_alt(color, n, means, covs, tau):
    """Оценка представления цвета"""
    if n == 0:
        return tau**2 + 1
    
    if n == 1:
        try:
            cov_inv = np.linalg.inv(covs[0])
            diff = color - means[0]
            mahal = diff.T @ cov_inv @ diff
            return mahal
        except np.linalg.LinAlgError:
            return tau**2 + 1
    
    # Для нескольких компонент
    scores = []
    
    # Одиночные компоненты
    for i in range(n):
        try:
            cov_inv = np.linalg.inv(covs[i])
            diff = color - means[i]
            mahal = diff.T @ cov_inv @ diff
            scores.append(mahal)
        except np.linalg.LinAlgError:
            scores.append(tau**2 + 1)
    
    # Парные комбинации
    for i in range(n):
        for j in range(i + 1, n):
            proj_cost = projected_cu_energy_alt(means[i], means[j], covs[i], covs[j], color)
            scores.append(proj_cost)
    
    return min(scores)

def get_voting_bin_alt(c):
    """Получает bin для голосования"""
    bin_idx = (np.floor(c[0] / 0.1), np.floor(c[1] / 0.1), np.floor(c[2] / 0.1))
    
    # Ограничиваем значения от 0 до 9
    bin_idx = tuple(min(int(b), 9) for b in bin_idx)
    
    return bin_idx

def kernel_values_alt(roi, img, m, n):
    """Создает guided filter kernel вокруг seed пикселя"""
    h_start, h_end = roi[0], roi[0] + roi[2]
    w_start, w_end = roi[1], roi[1] + roi[3]
    
    neighbourhood = img[w_start:w_end, h_start:h_end].copy()
    kernel = np.zeros((neighbourhood.shape[0], neighbourhood.shape[1], 3))
    kernel_weights = np.zeros((neighbourhood.shape[0], neighbourhood.shape[1]))
    
    S = img[n, m]  # Seed пиксель
    eps = 0.01
    
    for i in range(w_start, w_end):
        for j in range(h_start, h_end):
            # Создаем окно 20x20 вокруг пикселя
            patch_h_start = max(j - 10, 0)
            patch_h_end = min(j + 10, img.shape[1])
            patch_w_start = max(i - 10, 0)
            patch_w_end = min(i + 10, img.shape[0])
            
            patch = img[patch_w_start:patch_w_end, patch_h_start:patch_h_end]
            
            # Вычисляем среднее и стандартное отклонение для патча
            nMean = np.mean(patch, axis=(0, 1))
            nStddev = np.std(patch, axis=(0, 1)) + eps
            
            # Для каждого пикселя в окне вычисляем значение для ядра
            for k in range(patch_w_start, patch_w_end):
                for l in range(patch_h_start, patch_h_end):
                    x = k - (m - 10)
                    y = l - (n - 10)
                    
                    if 0 <= x < 20 and 0 <= y < 20:
                        I = img[k, l]
                        v1 = 1 + ((I[0] - nMean[0]) * (S[0] - nMean[0])) / (nStddev[0]**2)
                        v2 = 1 + ((I[1] - nMean[1]) * (S[1] - nMean[1])) / (nStddev[1]**2)
                        v3 = 1 + ((I[2] - nMean[2]) * (S[2] - nMean[2])) / (nStddev[2]**2)
                        
                        kernel[x, y] += np.array([v1, v2, v3])
            
            # Вычисляем вес ядра
            weight = np.sqrt(np.sum(kernel**2, axis=2))
            kernel_weights = weight
    
    return kernel_weights

def get_vote_alt(gradient, rep_score):
    """Voting energy - eq 10 unmixing paper"""
    gradient_norm = np.linalg.norm(gradient)
    vote = np.exp(-gradient_norm)**2 * (1 - np.exp(-rep_score))
    return vote

def next_bin(img, gradient, means, covs, votemask, tau):
    """Вычисляет следующий bin для seed пикселя"""
    votes = np.zeros((10, 10, 10))
    n = len(means)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if votemask[i, j] == 0:
                c = img[i, j]
                g = gradient[i, j]
                bin_idx = get_voting_bin_alt(c)
                
                score = representation_score_alt(c, n, means, covs, tau)
                
                if score > tau**2:
                    vote_val = get_vote_alt(g, score)
                    votes[bin_idx] += vote_val
                else:
                    votemask[i, j] = 255
    
    # Находим bin с наибольшим количеством голосов
    max_idx = np.unravel_index(np.argmax(votes), votes.shape)
    max_vote = votes[max_idx]
    
    return max_idx, max_vote

def get_next_seed_pixel_alt(img, gradient, means, covs, votemask, tau, bin_idx):
    """Находит следующий seed пиксель для цветовой модели"""
    low_b = bin_idx[0] * 0.1
    low_g = bin_idx[1] * 0.1
    low_r = bin_idx[2] * 0.1
    high_b = (bin_idx[0] + 1) * 0.1
    high_g = (bin_idx[1] + 1) * 0.1
    high_r = (bin_idx[2] + 1) * 0.1
    
    # Создаем маску для всех пикселей в bin
    mask = ((img[:, :, 0] >= low_b) & (img[:, :, 0] < high_b) &
            (img[:, :, 1] >= low_g) & (img[:, :, 1] < high_g) &
            (img[:, :, 2] >= low_r) & (img[:, :, 2] < high_r)).astype(np.uint8) * 255
    
    pixels_in_bin = np.where(mask[:, :, np.newaxis], img, 0)
    
    best_score = -1
    best_coords = (0, 0)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i, j] != 0 and votemask[i, j] != 255:
                grad = gradient[i, j]
                
                # Определяем окрестность 20x20
                roi_h_start = max(j - 10, 0)
                roi_h_end = min(j + 10, img.shape[1])
                roi_w_start = max(i - 10, 0)
                roi_w_end = min(i + 10, img.shape[0])
                
                neighbourhood = mask[roi_w_start:roi_w_end, roi_h_start:roi_h_end]
                
                # Подсчитываем пиксели в той же bin
                sp = np.sum(neighbourhood == 255)
                si = sp * np.exp(-np.linalg.norm(grad))
                
                if si >= best_score:
                    best_score = si
                    best_coords = (i, j)
    
    i, j = best_coords
    
    # Определяем окрестность для seed пикселя
    roi_h_start = max(j - 10, 0)
    roi_h_end = min(j + 10, img.shape[1])
    roi_w_start = max(i - 10, 0)
    roi_w_end = min(i + 10, img.shape[0])
    
    roi = (roi_h_start, roi_w_start, roi_h_end - roi_h_start, roi_w_end - roi_w_start)
    
    # Вычисляем значения ядра
    kernel = kernel_values_alt(roi, img, i, j)
    max_k = np.max(kernel)
    
    neighbourhood = img[roi_w_start:roi_w_end, roi_h_start:roi_h_end]
    neigh_mask = mask[roi_w_start:roi_w_end, roi_h_start:roi_h_end]
    
    # Собираем samples для вычисления ковариации
    samples = []
    for m in range(neighbourhood.shape[0]):
        for n in range(neighbourhood.shape[1]):
            if kernel[m, n] > (0.7 * max_k):
                v = neighbourhood[m, n]
                samples.append(v)
    
    if len(samples) == 0:
        votemask[:] = 255
        return
    
    samples_array = np.array(samples)
    
    # Вычисляем ковариацию
    mean_samp = np.mean(samples_array, axis=0)
    cov_samp = np.cov(samples_array.T, bias=False)
    
    # Добавляем небольшой шум для устойчивости
    cov_samp = cov_samp + np.eye(3) * 0.0001
    
    # Проверяем ковариацию
    try:
        cov_inv = np.linalg.inv(cov_samp)
        if not np.all(np.linalg.eigvals(cov_samp) > 0):
            cov_samp = np.eye(3) * 0.0001
    except np.linalg.LinAlgError:
        cov_samp = np.eye(3) * 0.0001
    
    # Добавляем новую компоненту в модель
    means.append(img[i, j])
    covs.append(cov_samp)

def get_global_color_model(image, means, covs, tau):
    """Вычисляет глобальную цветовую модель для изображения"""
    # Вычисляем градиент
    gradient = cv2.Laplacian(image, cv2.CV_64F)
    
    votemask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    vote = 0.0
    
    # Вычисляем первый bin
    bin_idx, vote = next_bin(image, gradient, means, covs, votemask, tau)
    
    # Итеративно добавляем цвета в модель
    count = 0
    min_pixels = (image.shape[0] * image.shape[1]) // 800
    
    while True:
        count = 0
        get_next_seed_pixel_alt(image, gradient, means, covs, votemask, tau, bin_idx)
        bin_idx, vote = next_bin(image, gradient, means, covs, votemask, tau)
        
        # Подсчитываем непредставленные пиксели
        count = np.sum(votemask == 0)
        
        # Условия остановки
        if count < min_pixels or vote <= 100.0:
            break
    
    return means, covs

# ==================== КЛАССЫ ИЗ MINIMIZATION.H ====================

def energy_func(v, means, covs, sparse=True):
    """Energy function - Eq. 4"""
    n = len(means)
    total_energy = 0.0
    
    alphas = v[:n]
    colors = v[n:].reshape(n, 3)
    
    for i in range(n):
        alpha = alphas[i]
        color = colors[i]
        
        try:
            cov_inv = np.linalg.inv(covs[i])
            diff = color - means[i]
            dist = diff.T @ cov_inv @ diff
            total_energy += alpha * dist
        except np.linalg.LinAlgError:
            total_energy += alpha * 1e6
    
    if sparse:
        sum_alpha = np.sum(alphas)
        sum_squared = np.sum(alphas**2)
        if sum_squared == 0:
            sparsity = 500.0
        else:
            sparsity = Constants.sigma * ((sum_alpha / sum_squared) - 1)
        total_energy += sparsity
    
    return total_energy

def g_constraint(v, n, color):
    """Constraint vector g - Eq. 4"""
    alphas = v[:n]
    colors = v[n:].reshape(n, 3)
    
    g1 = np.sum(alphas * colors[:, 0]) - color[0]
    g2 = np.sum(alphas * colors[:, 1]) - color[1]
    g3 = np.sum(alphas * colors[:, 2]) - color[2]
    g4 = np.sum(alphas) - 1
    
    return np.array([g1**2, g2**2, g3**2, g4**2])

def minimize_f_cg(x0, means, covs, color, lambda_vec, p, gt_alpha=None, refine=False):
    """Minimization using conjugate gradient"""
    n = len(means)
    
    def func(v):
        if refine:
            # Для refinement step
            energy_val = energy_func(v, means, covs, sparse=False)
            g_vec = g_constraint(v, n, color)
            if gt_alpha is not None:
                # Добавляем penalty для отклонения от gt_alpha
                alpha_diff = np.sum((v[:n] - gt_alpha)**2)
                energy_val += lambda_vec[3] * alpha_diff + 0.5 * p * alpha_diff**2
        else:
            energy_val = energy_func(v, means, covs, sparse=True)
            g_vec = g_constraint(v, n, color)
        
        penalty = lambda_vec @ g_vec + 0.5 * p * np.sum(g_vec**2)
        return energy_val + penalty
    
    # Простой градиентный спуск (упрощенная версия)
    x = x0.copy()
    alpha = 0.1
    max_iter = 100
    
    for iter in range(max_iter):
        # Центральная разность для градиента
        grad = np.zeros_like(x)
        eps = 1e-6
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        # Обновление
        x_new = x - alpha * grad
        
        # Ограничения
        x_new[:n] = np.clip(x_new[:n], 0, 1)
        x_new[n:] = np.clip(x_new[n:], 0, 1)
        
        # Проверка сходимости
        if np.linalg.norm(x_new - x) < Constants.tol:
            break
        
        x = x_new
    
    return x

def minimize_m_of_m(x0, means, covs, color, gt_alpha=None, refine=False):
    """Method of Multipliers"""
    p = 0.1
    lambda_vec = np.array([0.1, 0.1, 0.1, 0.1])
    n = len(means)
    
    x = x0.copy()
    iter = 0
    max_iter = 11
    
    while iter < max_iter:
        # Минимизация с фиксированными множителями
        x_new = minimize_f_cg(x, means, covs, color, lambda_vec, p, gt_alpha, refine)
        
        # Обновление множителей Лагранжа
        g_vec = g_constraint(x_new, n, color)
        lambda_vec = lambda_vec + p * g_vec
        
        # Обновление penalty параметра
        if np.linalg.norm(g_vec) > Constants.gamma * np.linalg.norm(g_constraint(x, n, color)):
            p = Constants.beta * p
        
        # Проверка сходимости
        if np.linalg.norm(x_new - x) < 0.0001:
            break
        
        x = x_new
        iter += 1
    
    return x

# ==================== КЛАСС PIXEL ====================

class Pixel:
    def __init__(self, color, coord):
        self.color = color
        self.coord = coord
    
    def min_index(self, color, means, covs):
        """Находит индекс ближайшего среднего"""
        n = len(means)
        distances = []
        
        for i in range(n):
            try:
                cov_inv = np.linalg.inv(covs[i])
                diff = color - means[i]
                dist = diff.T @ cov_inv @ diff
                distances.append(dist)
            except np.linalg.LinAlgError:
                distances.append(1e6)
        
        return np.argmin(distances)
    
    def unmix(self, means, covs, x_init=None):
        """Unmixing step"""
        n = len(means)
        
        if x_init is None or len(x_init) == 0:
            # Инициализация
            x = np.zeros(4 * n)
            
            for i in range(n):
                x[n + 3*i] = means[i][0]
                x[n + 3*i + 1] = means[i][1]
                x[n + 3*i + 2] = means[i][2]
            
            # Находим ближайший цвет
            mI = self.min_index(self.color, means, covs)
            x[mI] = 1.0
            x[n + 3*mI] = self.color[0]
            x[n + 3*mI + 1] = self.color[1]
            x[n + 3*mI + 2] = self.color[2]
        else:
            x = x_init.copy()
        
        # Минимизация
        x_min = minimize_m_of_m(x, means, covs, self.color)
        
        # Форматирование результатов
        alphas = x_min[:n]
        colors = x_min[n:].reshape(n, 3)
        
        return {
            'alphas': alphas,
            'colors': colors,
            'coords': self.coord
        }
    
    def refine(self, means, covs, gt_alpha):
        """Refinement step"""
        n = len(means)
        
        # Начинаем с предсказанных альфа
        x = np.zeros(4 * n)
        x[:n] = gt_alpha[:n]
        
        for i in range(n):
            x[n + 3*i] = means[i][0]
            x[n + 3*i + 1] = means[i][1]
            x[n + 3*i + 2] = means[i][2]
        
        # Минимизация для refinement
        x_min = minimize_m_of_m(x, means, covs, self.color, gt_alpha[:n], refine=True)
        
        # Форматирование результатов
        alphas = x_min[:n]
        colors = x_min[n:].reshape(n, 3)
        
        return {
            'alphas': alphas,
            'colors': colors,
            'coords': self.coord
        }

# ==================== GUIDED FILTER ====================

def boxfilter(I, r):
    """Box filter"""
    return cv2.blur(I, (r, r))

def guided_filter(I, p, r, eps, depth=-1):
    """Guided filter implementation"""
    if I.dtype != np.float32:
        I = I.astype(np.float32)
    if p.dtype != np.float32:
        p = p.astype(np.float32)
    
    # Размеры окна
    win_size = 2 * r + 1
    
    # Средние значения
    mean_I = boxfilter(I, r) / (win_size * win_size)
    mean_p = boxfilter(p, r) / (win_size * win_size)
    
    # Ковариации
    corr_I = boxfilter(I * I, r) / (win_size * win_size)
    corr_Ip = boxfilter(I * p, r) / (win_size * win_size)
    
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    
    # Коэффициенты
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # Средние коэффициенты
    mean_a = boxfilter(a, r) / (win_size * win_size)
    mean_b = boxfilter(b, r) / (win_size * win_size)
    
    # Результат
    q = mean_a * I + mean_b
    
    if depth != -1:
        if depth == np.uint8:
            q = np.clip(q, 0, 255).astype(np.uint8)
    
    return q

def matte_regularization(radius, guide_img, layers):
    """Matte Regularization Step"""
    n = len(layers)
    filtered_layers = []
    
    r = radius
    eps = 0.001 * 255 * 255
    
    sum_filtered_alphas = np.zeros((guide_img.shape[0], guide_img.shape[1]), dtype=np.float32)
    
    for i in range(n):
        layer = layers[i]
        
        # Извлекаем альфа-канал
        if layer.shape[2] == 4:
            alpha = layer[:, :, 3].astype(np.float32) / 255.0
        else:
            # Если нет альфа-канала, создаем бинарную маску
            alpha = np.any(layer != [255, 255, 255], axis=2).astype(np.float32)
        
        # Применяем guided filter
        guide_float = guide_img.astype(np.float32)
        alpha_filtered = guided_filter(guide_float, alpha, r, eps)
        
        sum_filtered_alphas += alpha_filtered
        filtered_layers.append(alpha_filtered)
    
    # Регуляризация альфа-значений
    regularized_layers = []
    
    for i in range(n):
        alpha_filtered = filtered_layers[i]
        
        # Нормализуем, чтобы сумма была 1
        eps = 1e-8
        reg_alpha = alpha_filtered / (sum_filtered_alphas + eps)
        
        # Создаем слой с цветом и альфа
        if layers[i].shape[2] == 4:
            color_layer = layers[i][:, :, :3]
        else:
            color_layer = layers[i]
        
        # Создаем RGBA слой
        rgba_layer = np.zeros((color_layer.shape[0], color_layer.shape[1], 4), dtype=np.uint8)
        rgba_layer[:, :, :3] = color_layer
        rgba_layer[:, :, 3] = (reg_alpha * 255).astype(np.uint8)
        
        regularized_layers.append(rgba_layer)
    
    return regularized_layers

# ==================== ОСНОВНАЯ ФУНКЦИЯ ДЛЯ SOFT COLOR SEGMENTATION ====================

def soft_color_segmentation(image, tau=0.1, max_colors=8):
    """
    Основная функция Soft Color Segmentation
    image: изображение в формате BGR (OpenCV)
    tau: параметр представления
    max_colors: максимальное количество цветов
    """
    # Конвертируем в double и нормализуем
    img_float = image.astype(np.float64) / 255.0
    
    # Инициализируем цветовую модель
    means = []
    covs = []
    
    # Шаг 1: Вычисление глобальной цветовой модели
    means, covs = get_global_color_model(img_float, means, covs, tau)
    
    n = len(means)
    print(f"Found {n} colors in the model")
    
    # Шаг 2: Sparse Color Unmixing
    rows, cols = image.shape[:2]
    layers = [np.zeros((rows, cols, 4), dtype=np.uint8) for _ in range(n)]
    
    # Проходим по всем пикселям
    for i in range(rows):
        for j in range(cols):
            color = img_float[i, j]
            pixel = Pixel(color, (j, i))
            
            # Unmixing
            result = pixel.unmix(means, covs)
            
            # Заполняем слои
            for k in range(n):
                alpha = result['alphas'][k] * 255
                layer_color = (result['colors'][k] * 255).astype(int)
                
                layers[k][i, j, :3] = layer_color
                layers[k][i, j, 3] = alpha
    
    # Шаг 3: Matte Regularization
    radius = max(1, int(60.0 / math.sqrt(1000000.0 / (cols * rows))))
    print(f"Filter radius: {radius}")
    
    regularized_layers = matte_regularization(radius, image, layers)
    
    # Шаг 4: Color Refinement
    refined_layers = [np.zeros((rows, cols, 4), dtype=np.uint8) for _ in range(n)]
    
    for i in range(rows):
        for j in range(cols):
            color = img_float[i, j]
            pixel = Pixel(color, (j, i))
            
            # Получаем gt_alpha из regularized layers
            gt_alpha = np.array([layer[i, j, 3] / 255.0 for layer in regularized_layers])
            
            # Refinement
            result = pixel.refine(means, covs, gt_alpha)
            
            # Заполняем слои
            for k in range(n):
                alpha = result['alphas'][k] * 255
                layer_color = (result['colors'][k] * 255).astype(int)
                
                refined_layers[k][i, j, :3] = layer_color
                refined_layers[k][i, j, 3] = alpha
    
    return refined_layers

# ==================== СТАРЫЙ КОД ДЛЯ ДЕКОМПОЗИЦИИ ====================

# Проверка наличия модели для старого метода
def check_model_exists():
    """Проверяет наличие модели в папке model/"""
    model_path = Path("model/mask_generator7.pth")
    
    if model_path.exists():
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        return True, f"✅ Модель найдена: {model_path} ({file_size:.2f} MB)"
    else:
        return False, "❌ Модель не найдена в папке model/"

model_available, model_message = check_model_exists()

# Классы для старого метода DECOMPOSE
class _MyDataset(torch.utils.data.Dataset):
    def __init__(self, img, num_primary_color, palette):
        self.img = img.convert("RGB")
        self.palette_list = palette.reshape(-1, num_primary_color * 3)
        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        np_img = np.array(self.img)
        np_img = np_img.transpose((2, 0, 1))
        target_img = np_img / 255  # 0~1

        primary_color_layers = self._make_primary_color_layers(
            self.palette_list[index], target_img
        )

        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color_layers.astype(np.float32))

        return target_img, primary_color_layers

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
        in_dim = 3 + num_primary_color * 3
        out_dim = num_primary_color

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

        h1 = self.bn1(F.relu(self.conv1(x)))
        h2 = self.bn2(F.relu(self.conv2(h1)))
        h3 = self.bn3(F.relu(self.conv3(h2)))
        h4 = self.bnde1(F.relu(self.deconv1(h3)))
        h4 = torch.cat((h4, h2), 1)
        h5 = self.bnde2(F.relu(self.deconv2(h4)))
        h5 = torch.cat((h5, h1), 1)
        h6 = self.bnde3(F.relu(self.deconv3(h5)))
        h6 = torch.cat((h6, target_img), 1)
        h7 = self.bn4(F.relu(self.conv4(h6)))

        return torch.sigmoid(self.conv5(h7))

# Функции для старого метода
def get_dominant_colors(img: Image.Image, num_colors: int) -> list[tuple]:
    """Получение доминирующих цветов из изображения с использованием K-means"""
    img_array = np.array(img)
    
    if img.mode == "RGBA":
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_colors = colors[sorted_indices]
    
    return [tuple(color) for color in sorted_colors]

def decompose_fast_soft_color(
    input_image: Image.Image,
    num_colors: int = 7,
    palette: list[tuple] = None,
    resize_scale_factor: float = 1.0
) -> list[Image.Image]:
    """Старая функция для разложения изображения с использованием нейронной сети"""
    layersRGBA = []
    
    if not model_available:
        st.error("Модель не найдена. Невозможно выполнить метод Decompose.")
        return []
    
    if num_colors < 2 or num_colors > 8:
        st.error(f"Количество цветов должно быть от 2 до 8. Получено: {num_colors}")
        return []
    
    if palette is None:
        palette = get_dominant_colors(input_image, num_colors)
    else:
        if len(palette) != num_colors:
            while len(palette) < num_colors:
                palette.append(palette[-1] if palette else (128, 128, 128))
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
        mask_generator = _MaskGeneratorModel(num_colors).to(cpu)
        
        model_path = Path("model/mask_generator7.pth")
        mask_generator.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
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
                processed_alpha_layers = alpha_normalize(processed_alpha_layers)
                
                mono_RGBA_layers = torch.cat(
                    (primary_color_layers, processed_alpha_layers), dim=2
                )
                
                mono_RGBA_layers = mono_RGBA_layers[0]
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
    """Преобразует слои RGBA в формат BGR с прозрачностью"""
    cv_layers = []
    color_info_list = []
    
    for i, pil_layer in enumerate(decompose_layers):
        rgba_array = np.array(pil_layer)
        
        if rgba_array.shape[2] == 4:
            rgb_array = rgba_array[:, :, :3]
            alpha_array = rgba_array[:, :, 3] / 255.0
            
            layer_with_bg = np.zeros_like(rgb_array, dtype=np.uint8)
            
            for c in range(3):
                layer_with_bg[:, :, c] = rgb_array[:, :, c] * alpha_array + bg_color[c] * (1 - alpha_array)
            
            bgr_layer = cv2.cvtColor(layer_with_bg, cv2.COLOR_RGB2BGR)
            
            mask = alpha_array > 0.1
            if np.any(mask):
                masked_colors = rgb_array[mask]
                if len(masked_colors) > 0:
                    median_color = np.median(masked_colors, axis=0).astype(int)
                    median_color_bgr = (median_color[2], median_color[1], median_color[0])
                else:
                    median_color_bgr = bg_color
            else:
                median_color_bgr = bg_color
            
            coverage_percentage = (np.sum(mask) / mask.size) * 100
            
            cv_layers.append(bgr_layer)
            color_info_list.append({
                'color': median_color_bgr,
                'percentage': coverage_percentage
            })
        else:
            bgr_layer = cv2.cvtColor(rgba_array, cv2.COLOR_RGB2BGR)
            
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
    """Разделение цветов с использованием алгоритма K-means"""
    if n_colors < 2 or n_colors > 8:
        st.error(f"Количество цветов должно быть от 2 до 8. Получено: {n_colors}")
        return [], []
    
    try:
        pixels = img.reshape(-1, 3)
        
        if bg_color:
            bg_mask = np.all(pixels == bg_color, axis=1)
            if np.any(bg_mask):
                pixels = pixels[~bg_mask]
        
        if len(pixels) == 0:
            st.warning("Изображение состоит только из фона")
            return [], []
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        full_labels = np.zeros(img.shape[0] * img.shape[1], dtype=int) - 1
        if bg_color:
            bg_mask_full = np.all(img.reshape(-1, 3) == bg_color, axis=1)
            non_bg_indices = np.where(~bg_mask_full)[0]
            if len(non_bg_indices) >= len(labels):
                full_labels[non_bg_indices[:len(labels)]] = labels
        
        color_layers = []
        color_info = []
        
        for i in range(n_colors):
            mask = (full_labels == i).reshape(img.shape[0], img.shape[1])
            layer = np.full_like(img, bg_color)
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

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def convert_to_png(image_array, filename):
    """Конвертирует массив изображения в формат PNG"""
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_array)
        ax.axis('off')
        fig.tight_layout(pad=0)
        
        png_buffer = io.BytesIO()
        plt.savefig(png_buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)
        
        png_buffer.seek(0)
        return png_buffer.getvalue()
    except Exception as e:
        st.error(f"Ошибка при создании PNG: {e}")
        return None

def create_bw_mask(layer, bg_color):
    """Создает черно-белую маску из цветного слоя"""
    is_background = np.all(layer == bg_color, axis=2)
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

# ==================== КОНФИГУРАЦИЯ СТРАНИЦЫ STREAMLIT ====================

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
        methods = ["K-средних кластеризация", "Soft Color Segmentation (Unmixing)"]
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
        
        # Параметры для Soft Color Segmentation
        if selected_method == "Soft Color Segmentation (Unmixing)":
            st.markdown("<h4>⚙️ Параметры Unmixing</h4>", unsafe_allow_html=True)
            tau_param = st.slider("Параметр tau", 0.01, 0.5, 0.1, 0.01,
                                 help="Порог представления цвета (меньше = больше цветов)",
                                 label_visibility="collapsed")
        
        # Дополнительные настройки для нейронной сети
        if selected_method == "Fast Soft Color Segmentation (нейронная сеть)" and model_available:
            st.markdown("<h4>⚡ Настройки нейронной сети</h4>", unsafe_allow_html=True)
            resize_factor = st.slider("Масштаб", 0.5, 2.0, 1.0, 0.1,
                                     help="Коэффициент изменения размера для обработки",
                                     label_visibility="collapsed")
        
        # Дополнительные опции
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
    
    with col2:
        st.markdown("<h3 class='sub-header'>🎨 Разделенные цветовые слои</h3>", unsafe_allow_html=True)
        
        # Кнопка для запуска обработки
        if st.button("🚀 Начать разделение цветов", type="primary", use_container_width=True):
            with st.spinner("🔄 Обработка изображения... Пожалуйста, подождите."):
                try:
                    progress_bar = st.progress(0)
                    
                    if selected_method == "K-средних кластеризация":
                        progress_bar.progress(20)
                        # Используем K-means
                        color_layers, color_info = kmeans_color_separation(
                            img_cv, 
                            n_colors=num_colors,
                            bg_color=bg_color_rgb
                        )
                        progress_bar.progress(100)
                    
                    elif selected_method == "Soft Color Segmentation (Unmixing)":
                        progress_bar.progress(10)
                        st.info("⚠️ Внимание: Этот метод может работать медленно для больших изображений")
                        
                        # Используем новый алгоритм Soft Color Segmentation
                        # Преобразуем в формат для алгоритма
                        if 'tau_param' in locals():
                            tau_value = tau_param
                        else:
                            tau_value = 0.1
                        
                        progress_bar.progress(30)
                        
                        # Выполняем soft color segmentation
                        try:
                            soft_layers = soft_color_segmentation(img_cv, tau=tau_value, max_colors=num_colors)
                            progress_bar.progress(70)
                            
                            # Преобразуем слои в формат для отображения
                            color_layers = []
                            color_info = []
                            
                            for i, layer in enumerate(soft_layers):
                                # Извлекаем BGR цвет
                                bgr_layer = layer[:, :, :3]
                                
                                # Извлекаем альфа-канал
                                alpha_channel = layer[:, :, 3] / 255.0
                                
                                # Создаем слой с фоном
                                layer_with_bg = np.zeros_like(bgr_layer, dtype=np.uint8)
                                
                                for c in range(3):
                                    layer_with_bg[:, :, c] = bgr_layer[:, :, c] * alpha_channel + bg_color_rgb[c] * (1 - alpha_channel)
                                
                                color_layers.append(layer_with_bg)
                                
                                # Вычисляем информацию о цвете
                                mask = alpha_channel > 0.1
                                if np.any(mask):
                                    # Получаем цвета из оригинального слоя
                                    colors_in_layer = bgr_layer[mask]
                                    if len(colors_in_layer) > 0:
                                        # Используем медиану для лучшей оценки цвета
                                        median_color = np.median(colors_in_layer, axis=0).astype(int)
                                        dominant_color = (int(median_color[0]), int(median_color[1]), int(median_color[2]))
                                    else:
                                        dominant_color = bg_color_rgb
                                else:
                                    dominant_color = bg_color_rgb
                                
                                # Процент покрытия
                                coverage_percentage = (np.sum(mask) / mask.size) * 100
                                
                                color_info.append({
                                    'color': dominant_color,
                                    'percentage': coverage_percentage
                                })
                            
                            progress_bar.progress(90)
                            
                        except Exception as e:
                            st.error(f"Ошибка в Soft Color Segmentation: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                            color_layers, color_info = [], []
                        
                        progress_bar.progress(100)
                    
                    elif selected_method == "Fast Soft Color Segmentation (нейронная сеть)":
                        progress_bar.progress(20)
                        # Используем нейронную сеть
                        if not model_available:
                            st.error("Модель не найдена. Используйте другой метод.")
                            color_layers, color_info = [], []
                        else:
                            palette_colors = get_dominant_colors(image, num_colors)
                            progress_bar.progress(40)
                            
                            decompose_layers = decompose_fast_soft_color(
                                image,
                                num_colors=num_colors,
                                palette=palette_colors,
                                resize_scale_factor=resize_factor if 'resize_factor' in locals() else 1.0
                            )
                            progress_bar.progress(70)
                            
                            if decompose_layers:
                                color_layers, color_info = decompose_layers_to_cv_format(
                                    decompose_layers, 
                                    bg_color_rgb
                                )
                                progress_bar.progress(100)
                            else:
                                st.error("Не удалось выполнить разделение с помощью нейронной сети.")
                                color_layers, color_info = [], []
                    
                    # Сохраняем результаты в session state
                    st.session_state.color_layers = color_layers
                    st.session_state.color_info = color_info
                    
                    if color_layers and color_info:
                        st.success(f"✅ Успешно создано {len(color_layers)} цветовых слоев!")
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
                        
                        st.markdown(f"""
                        <div style='padding: 15px; background-color: #f8f9fa; border-radius: 10px;'>
                            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                                <div class='color-chip' style='background-color: {hex_color};'></div>
                                <div>
                                    <strong style='font-size: 1.2em;'>{hex_color}</strong><br>
                                    <span style='color: #666; font-size: 0.9em;'>Цвет слоя</span>
                                </div>
                            </div>
                            <div style='margin-bottom: 10px;'>
                                <strong>RGB:</strong> {info['color'][::-1]}<br>
                                <strong>Покрытие:</strong> {info['percentage']:.1f}%<br>
                                <strong>Пикселей:</strong> {layer.shape[1]} × {layer.shape[0]}
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
                        st.markdown(f"""
                        <div style='display: flex; align-items: center; padding: 8px; background-color: {'#e8f5e9' if visibility else '#f5f5f5'}; border-radius: 5px;'>
                            <div style='width: 25px; height: 25px; background-color: {hex_color}; border: 1px solid #000; border-radius: 4px; margin-right: 10px;'></div>
                            <div>
                                <div><strong>Слой {i+1}</strong></div>
                                <div style='font-size: 0.8em; color: #666;'>{hex_color} • {color_info[i]['percentage']:.1f}%</div>
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
                        if 'combined_png_data' in locals() and combined_png_data:
                            combined_path = os.path.join(tmpdirname, "combined_mask.png")
                            with open(combined_path, 'wb') as f:
                                f.write(combined_png_data)
                            all_files.append(combined_path)
                        
                        if 'combined_color_png' in locals() and combined_color_png:
                            combined_color_path = os.path.join(tmpdirname, "combined_preview.png")
                            with open(combined_color_path, 'wb') as f:
                                f.write(combined_color_png)
                            all_files.append(combined_color_path)
                        
                        # Создаем README файл
                        readme_content = f"""# ColorSep Pro - Экспортированные слои

Дата создания: {time.strftime('%Y-%m-%d %H:%M:%S')}
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
                            readme_content += f"- Слой {i+1}: {hex_color}, RGB{info['color'][::-1]}, Покрытие: {info['percentage']:.1f}%\n"
                        
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

col_method1, col_method2, col_method3 = st.columns(3)

with col_method1:
    st.markdown("""
    <div class="method-card">
        <h4>🎯 K-средних кластеризация</h4>
        <p><strong>Описание:</strong> Классический алгоритм для группировки похожих цветов.</p>
        <p><strong>Преимущества:</strong></p>
        <ul>
            <li>Быстрая обработка</li>
            <li>Контролируемое количество цветов</li>
            <li>Хорошо работает с четкими цветами</li>
        </ul>
        <p><strong>Идеально для:</strong> Логотипы, векторная графика</p>
    </div>
    """, unsafe_allow_html=True)

with col_method2:
    st.markdown("""
    <div class="method-card">
        <h4>🧮 Soft Color Segmentation (Unmixing)</h4>
        <p><strong>Описание:</strong> Алгоритм на основе разложения цветов (unmixing).</p>
        <p><strong>Преимущества:</strong></p>
        <ul>
            <li>Точное разделение цветов</li>
            <li>Учитывает статистику цветов</li>
            <li>Поддерживает плавные переходы</li>
        </ul>
        <p><strong>Идеально для:</strong> Фотографии, градиенты, сложные изображения</p>
    </div>
    """, unsafe_allow_html=True)

with col_method3:
    if model_available:
        st.markdown("""
        <div class="method-card">
            <h4>⚡ Fast Soft Color Segmentation</h4>
            <p><strong>Описание:</strong> Нейронная сеть для продвинутого разделения цветов.</p>
            <p><strong>Преимущества:</strong></p>
            <ul>
                <li>Создает слои с прозрачностью</li>
                <li>Сохраняет плавные переходы</li>
                <li>Быстрая обработка с GPU</li>
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

# ==================== ФУТЕР ====================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 30px; background-color: #f8f9fa; border-radius: 10px;">
    <h4>🎨 ColorSep Pro v3.0</h4>
    <p>Профессиональный инструмент для разделения цветов с алгоритмом Unmixing</p>
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
        st.write(f"**NumPy:** {np.__version__}")
        st.write(f"**Streamlit:** {st.__version__}")
        
        if torch.cuda.is_available():
            st.write(f"**GPU Память:** {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Статус модели
        if model_available:
            st.success("✅ Модель нейронной сети загружена")
        else:
            st.warning("⚠️ Модель нейронной сети не найдена")
            
except Exception as e:
    st.sidebar.error(f"Ошибка проверки системы: {e}")

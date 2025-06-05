from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def determine_optimal_clusters_elbow(latents: np.ndarray, max_k: int = 50) -> int:
    """Определить оптимальное количество кластеров, используя метод локтя."""
    n_samples = latents.shape[0]
    max_k = min(max_k, n_samples // 2)
    
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(latents)
        inertias.append(kmeans.inertia_)
    
    # Найдти "локоть" с помощью второй производной
    deltas = np.diff(inertias)
    delta2 = np.diff(deltas)
    optimal_k = np.argmin(delta2) + 2  # +2 из-за второй производной (каждая удаляет по одному элементу)
    
    return max(2, optimal_k)

def cluster_latent_vectors(images_and_latents: List[Tuple[str, np.ndarray]]) -> \
        Tuple[str, Tuple[Tuple[str, float, bool], ...]]:
    """
    Кластеризовать латентные векторы с использованием метода адаптивного расстояния с запеченными параметрами.
    
    Входные данные:
        images_and_latents: Список из кортежей (имя файла, латентный вектор)
    
    Выходные данные:
        Кластеры: (имя кластера, ((имя файла, вероятность, входит ли в кластер), ...))
    """
    
    # Разделение латентов и имён файлов
    filenames = [item[0] for item in images_and_latents]
    latents = np.array([item[1] for item in images_and_latents])
    
    # Сдандартизация латентных векторов
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)
    
    # Расчёт оптимального количества k с помощью метода локтя
    optimal_k = determine_optimal_clusters_elbow(latents_scaled, max_k=20)
    
    # K-means центры коастеров
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(latents_scaled)
    centers = kmeans.cluster_centers_
    
    # Расчёт расстояний от центров кластеров каждой точки
    distances = np.zeros((len(latents_scaled), optimal_k))
    for i, center in enumerate(centers):
        distances[:, i] = np.linalg.norm(latents_scaled - center, axis=1)
    
    # Расчёт адаптивного порог для каждого кластера (75-ый процентиль)
    cluster_thresholds = []
    for cluster_id in range(optimal_k):
        cluster_distances = distances[:, cluster_id]
        threshold = np.percentile(cluster_distances, 75)  # distance_percentile=75
        cluster_thresholds.append(threshold)
    
    # Преобразование расстояний в вероятности с помощью адаптивных порогов
    probabilities = np.zeros_like(distances)
    temperature = 2.0  # Пока самая эффективная температура
    
    for cluster_id in range(optimal_k):
        cluster_distances = distances[:, cluster_id]
        threshold = cluster_thresholds[cluster_id]
        
        # Вероятность с экспоненциальым затуханием
        normalized_distances = (cluster_distances / threshold) ** 2
        probabilities[:, cluster_id] = np.exp(-normalized_distances / temperature)
    
    # Организация кластеров (список)
    clusters = []
    for cluster_id in range(optimal_k):
        cluster_members = []
        cluster_probs = probabilities[:, cluster_id]
        
        # Порог: среднее + 0.5 * стандартное отклонение
        prob_mean = np.mean(cluster_probs)
        prob_std = np.std(cluster_probs)
        prob_threshold = max(0.1, prob_mean + 0.5 * prob_std)
        
        for i, filename in enumerate(filenames):
            prob = float(cluster_probs[i])
            is_member = prob >= prob_threshold
            cluster_members.append((filename, prob, is_member))
        
        # Сопртировка по самым вероятным значениям
        cluster_members.sort(key=lambda x: -x[1])
        clusters.append(("", tuple(cluster_members)))  # Имя (пустое) и расчёты
    
    return clusters

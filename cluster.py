# Информация о ВКР "Clagnosco":
#  ФИО автора: Погорельских Константин Владимирович
#  Тема ВКР: «Классификация изображений с помощью искусственного интеллекта (на примере Частного образовательного учреждения высшего образования «Московский университет имени С.Ю. Витте»).»
#  ВУЗ: ЧОУ ВО «Московский университет им. С.Ю. Витте»
#  Специальность: Прикладная информатика [09.03.03] Бакалавр
#  Факультет: Информационных технологий
#  Специализация / Профиль подготовки: Искусственный интеллект и анализ данных
#  Учебная группа: ИД 23.3/Б3-21

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def determine_optimal_clusters_elbow(latents: np.ndarray) -> int:
    """Определить оптимальное количество кластеров, используя метод локтя."""
    n_samples = len(latents)
    if n_samples < 3:
        return 2
    
    max_k = min(len(latents), (len(latents) // 5) + 2, 100)
    k_range = range(2, max_k + 1)
    
    inertias = []
    
    for k in tqdm(k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(latents)
        inertias.append(kmeans.inertia_)

    knee = KneeLocator(list(k_range), inertias, curve="convex", direction="decreasing")
    optimal_k = knee.elbow

    if optimal_k is None:
        optimal_k = 2
    return optimal_k


def cluster_latent_vectors(images_and_latents: List[Tuple[str, np.ndarray]],
                           cluster_amount=-1,
                           print_process=False) -> \
        Tuple[str, Tuple[Tuple[str, float, bool], ...]]:
    """
    Кластеризовать латентные векторы с использованием метода адаптивного расстояния с запеченными параметрами.
    
    Входные данные:
        images_and_latents: Список из кортежей (имя файла, латентный вектор)
        cluster_amount: Количество кластеров (-1 по умолчанию означает, что количество кластеров будет расчитано методом локтя)
        print_process: bool (по умолчанию False) -- печатает в консоль процесс выполнения
    
    Выходные данные:
        Кластеры: (имя кластера, ((имя файла, вероятность, входит ли в кластер), ...))
    """
    
    print("Начало кластеризации") if print_process else None

    # Разделение латентов и имён файлов
    filenames = [item[0] for item in images_and_latents]
    latents = np.array([item[1] for item in images_and_latents])
    
    # Сдандартизация латентных векторов
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)
    
    # Расчёт оптимального количества k с помощью метода локтя при cluster_amount == -1
    if cluster_amount == -1:
        optimal_k = determine_optimal_clusters_elbow(latents_scaled)
        print(f"optimal_k (Расчёт оптимального количества k с помощью метода локтя) завершено: {optimal_k}") if print_process else None
    elif cluster_amount > 1 and len(latents) > 1:
        optimal_k = min(cluster_amount, len(latents))
        print(f"optimal_k был выбран вручную: {optimal_k}") if print_process else None
    else:
        raise ValueError("Слишком малое количество кластеров.")
    
    # K-means центры кластеров
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(latents_scaled)
    centers = kmeans.cluster_centers_
    print("kmeans (K-means центры кластеров) завершено") if print_process else None
    
    # Расчёт расстояний от центров кластеров каждой точки
    distances = np.zeros((len(latents_scaled), optimal_k))
    for i, center in enumerate(centers):
        distances[:, i] = np.linalg.norm(latents_scaled - center, axis=1)
    print("distances (Расчёт расстояний от центров кластеров каждой точки) завершено") if print_process else None
    
    # Расчёт адаптивного порога для каждого кластера (75-ый процентиль)
    cluster_thresholds = []
    for cluster_id in range(optimal_k):
        cluster_distances = distances[:, cluster_id]
        threshold = np.percentile(cluster_distances, 75)  # distance_percentile=75
        cluster_thresholds.append(threshold)
    print("cluster_thresholds (Расчёт адаптивного порога для каждого кластера) завершено") if print_process else None
    
    # Преобразование расстояний в вероятности с помощью адаптивных порогов
    probabilities = np.zeros_like(distances)
    temperature = 2.0  # Пока самая эффективная температура
    
    for cluster_id in range(optimal_k):
        cluster_distances = distances[:, cluster_id]
        threshold = cluster_thresholds[cluster_id]
        
        # Вероятность с экспоненциальным затуханием
        normalized_distances = (cluster_distances / threshold) ** 2
        probabilities[:, cluster_id] = np.exp(-normalized_distances / temperature)
    print("probabilities (Преобразование расстояний в вероятности с помощью адаптивных порогов) завершено") if print_process else None
    
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
            is_member = bool(prob >= prob_threshold)
            cluster_members.append((filename, prob, is_member))
        
        # Сопртировка по самым вероятным значениям
        cluster_members.sort(key=lambda x: -x[1])
        clusters.append(("", tuple(cluster_members)))  # Имя (пустое) и расчёты

        print(f"cluster_members (Организация кластера {cluster_id+1}/{optimal_k}) завершено") if print_process else None
    
    return clusters

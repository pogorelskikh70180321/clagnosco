from autoencoder import *
import matplotlib.pyplot as plt
import pickle


# Информация о проекте:
#  ФИО автора: Погорельских Константин Владимирович
#  Тема ВКР: «Классификация изображений с помощью искусственного интеллекта (на примере Частного образовательного учреждения высшего образования «Московский университет имени С.Ю. Витте»).»
#  ВУЗ: ЧОУ ВО «Московский университет им. С.Ю. Витте»
#  Специальность: Прикладная информатика [09.03.03] Бакалавр
#  Факультет: Информационных технологий
#  Специализация / Профиль подготовки: Искусственный интеллект и анализ данных
#  Учебная группа: ИД 23.3/Б3-21


# Загрузка изображений (28.5 MB)
url = "https://huggingface.co/pogorzelskich/clagnosco_2025-05-11/resolve/main/images_origin.pkl?download=true"
response = requests.get(url)
response.raise_for_status()
images_origin = pickle.loads(response.content)

# Загрузка модели (1.23 GB)
url = "https://huggingface.co/pogorzelskich/clagnosco_2025-05-11/resolve/main/model.pt?download=true"
model, _ = model_loader(url)
model.eval()

images_reconstructed = []
for img in images_origin:
    images_reconstructed.append(run_image_through_autoencoder(model, img)[1])

# PLT Оригиналы и реконструкции
img_count = 12
fig, axs = plt.subplots(2, img_count, figsize=(img_count * 2, 4))
for i in range(img_count):
    axs[0, i].imshow(images_origin[i], cmap='gray')
    axs[0, i].axis('off')
    axs[1, i].imshow(images_reconstructed[i], cmap='gray')
    axs[1, i].axis('off')
plt.show()

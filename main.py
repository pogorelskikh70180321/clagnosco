from autoencoder import *
import matplotlib.pyplot as plt
import pickle


if __name__ == "__main__":
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
    fig, axs = plt.subplots(2, img_count, figsize=(img_count*2, 4))
    for i in range(img_count):
        axs[0, i].imshow(images_origin[i], cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(images_reconstructed[i], cmap='gray')
        axs[1, i].axis('off')
    plt.show()

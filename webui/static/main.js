var safeReload = true;
var loggingFetch = true;
var serverConnect = true;
var serverReconnect = false;
var reloadNeeded = false;

var clagnoscoClassesSizes = [];
var clagnoscoImagesNames = [];
var imagesFolder;
const imgView = '/img/';
const smallImgView = '/img_small/';


function setupReloadConfirmation() {
    const hiddenForm = document.createElement('form');
    hiddenForm.classList.add("hidden");
    hiddenForm.innerHTML = '<input type="submit">';
    document.getElementsByClassName('container')[0].appendChild(hiddenForm);
    
    hiddenForm.addEventListener('submit', function(e) {
        e.preventDefault();
    });
    
    window.addEventListener('beforeunload', function (e) {
        if (!reloadNeeded) {
            if (!safeReload) {
                e.preventDefault();
                hiddenForm.querySelector('input[type="submit"]').click();
            }
        }
    });
}

function showCustomAlert(message, timeout=5000) {
    const alertBox = document.getElementById("custom-alert");
    alertBox.textContent = message;
    alertBox.classList.add("show");

    if (timeout !== -1) {
        setTimeout(() => {
            alertBox.classList.remove("show");
        }, timeout);
    }
}

async function sendToServer(data, isBasic=false) {
    try {
        if (loggingFetch) {
            console.log("Запрос:", data);
        }

        const response = await fetch('/fetch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (reloadNeeded) {
            safeReload = true;
            serverConnect = true;
            serverReconnect = true;
            showCustomAlert("Соединение с сервером восстановлено. Перезагрузите страницу.", timeout=-1);
            return null;
        }

        if (!response.ok) {
            throw new Error('Нет ответа');
        }

        const answer = await response.json();
        if (loggingFetch) {
            console.log("Ответ:", answer);
        }
        
        return answer;

    } catch (error) {
        if (!isBasic) {
            console.error('Ошибка:', error);
            const answer = await sendToServer({ command: "basicResponse" }, true);
            
            if (answer["status"] === "basicResponseSuccess") {
                if (!serverConnect) {
                    reloadNeeded = true;
                    safeReload = true;
                    if (!serverReconnect) {
                        showCustomAlert("Соединение с сервером восстановлено. Перезагрузите страницу.", timeout=-1);
                    }
                }
            } else {
                if (serverConnect) {
                    reloadNeeded = true;
                    serverConnect = false;
                    serverReconnect = false;
                    safeReload = true;
                    if (!serverReconnect) {
                        showCustomAlert("Потеряно соединение с сервером. Процесс не сохранён.", timeout=-1);
                    }
                }
            }
        } else {
            if (serverConnect) {
                reloadNeeded = true;
                serverConnect = false;
                serverReconnect = false;
                safeReload = true;
                if (!serverReconnect) {
                    showCustomAlert("Потеряно соединение с сервером. Процесс не сохранён.", timeout=-1);
                }
            }
        }
        return { error: error.message };
    }
}

function clearCache(confirmClearingCache=true) {
    let imgDir = document.getElementById("localFolder");
    
    if (confirmClearingCache) {
        const confirmStatus = window.confirm(`Очистить кэш в директории «${imgDir.value}»?`);

        if (!confirmStatus) {
            return null;
        }
    }
    let instruction = {'command': 'clearCache',
                       'imgDir': imgDir.value};

    sendToServer(instruction).then(answer => {
        if (answer["status"] === "cacheCleared") {
            showCustomAlert("Кэш успешно очищен в выбранной директории", timeout=5000);
        } else if (answer["status"] === "error") {
            alert(answer["message"]);
        } else {
            // alert("Странный ответ сервера.");
        }
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
        resetAll();
    });
}

function resetAll(confirmResetAll=true) {
    if (reloadNeeded) {
        confirmResetAll=false;
    }

    if (confirmResetAll) {
        const confirmStatus = window.confirm(`Сбросить всё?`);

        if (!confirmStatus) {
            return null;
        }
    }

    let loading = document.getElementsByClassName('fullscreen-loader')[0];
    loading.classList.remove('hidden');

    let imgDir = document.getElementById("localFolder");
    let modelName = document.getElementById("modelNameSelect");
    let caching = document.getElementById("caching");

    imgDir.disabled = false;
    modelName.disabled = false;  
    caching.disabled = false;


    let menuStatusInit = document.getElementById("menuStatusInit");
    let menuStatusProcessing = document.getElementById("menuStatusProcessing");
    let menuStatusDone = document.getElementById("menuStatusDone");
    
    menuStatusInit.classList.remove("hidden");
    menuStatusProcessing.classList.add("hidden");
    menuStatusDone.classList.add("hidden");

    clagnoscoClassesSizes = [];
    clagnoscoImagesNames = [];
    imagesFolder;

    deleteAllImages();
    deleteAllClagnoscoClasses(false);
    populateModels();

    document.getElementsByClassName('classes-tab')[0].classList.add('hidden');
    document.getElementById('classTab').classList.add('hidden');

    let classTabLoading = document.getElementsByClassName("class-tab-loading")[0];
    classTabLoading.classList.add('hidden');

    safeReload = true;

    loading.classList.add('hidden');
}


function launchProcessing() {
    safeReload = false;

    let imgDir = document.getElementById("localFolder");
    let modelName = document.getElementById("modelNameSelect");
    let caching = document.getElementById("caching");
    let instruction = {'command': 'launchProcessing',
                       'imgDir': imgDir.value,
                       'modelName': modelName.value,
                       'caching': caching.checked};
    imgDir.disabled = true;
    modelName.disabled = true;  
    caching.disabled = true;

    let menuStatusInit = document.getElementById("menuStatusInit");
    let menuStatusProcessing = document.getElementById("menuStatusProcessing");
    let menuStatusDone = document.getElementById("menuStatusDone");
    
    if (modelName.value == "download") {
        menuStatusProcessing.children[1].textContent = "Загрузка модели из интернета...";
    } else {
        menuStatusProcessing.children[1].textContent = "Загрузка модели...";
    }

    menuStatusInit.classList.add("hidden");
    menuStatusProcessing.classList.remove("hidden");
    menuStatusDone.classList.add("hidden");

    sendToServer(instruction).then(answer => {
        if (answer["status"] === "readyToCluster") {
            clusterImages();
        } else if (answer["status"] === "error") {
            alert(answer["message"]);
            
            resetAll();
        } else {
            // alert("Странный ответ сервера.");
            resetAll();
        }
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
        resetAll();
    });
}

function clusterImages(imagesCount=-1) {
    let menuStatusProcessingText = document.getElementById("menuStatusProcessing").children[1];
    if (imagesCount === -1) {
        menuStatusProcessingText.textContent = "Обработка изображений...";
    } else {
        menuStatusProcessingText.textContent = `Обработка изображений (${imagesCount})...`;
    }

    let instruction = {'command': 'clusterImages'};
    
    sendToServer(instruction).then(answer => {
        if (answer["status"] === "readyToPopulate") {
            // alert("Processing launched successfully.");
            // console.log(answer);
            
            clagnoscoClassesSizes = answer["classesSizes"];
            clagnoscoImagesNames = answer["imagesNames"];
            imagesFolder = answer["imagesFolder"];
            
            menuStatusProcessingText.textContent = "Распределение кластеров...";

            showCustomAlert(`Все изображения обработаны (${clagnoscoImagesNames.length}). Было найдено следующее количество кластеров: ${clagnoscoClassesSizes.length}`);
            populateClagnoscoClasses();
            populateImages();

            let menuStatusInit = document.getElementById("menuStatusInit");
            let menuStatusProcessing = document.getElementById("menuStatusProcessing");
            let menuStatusDone = document.getElementById("menuStatusDone");

            menuStatusInit.classList.add("hidden");
            menuStatusProcessing.classList.add("hidden");
            menuStatusDone.classList.remove("hidden");

        } else if (answer["status"] === "error") {
            alert(answer["message"]);
            resetAll();
        } else {
            // alert("Странный ответ сервера.");
            resetAll();
        }
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
        resetAll();
    });
}

function bracketsRemoval(text) {
    return text.split("(")[1].split(")")[0];
}

function isLocalServerURL(urlString) {
  try {
    const url = new URL(urlString);
    const hostname = url.hostname;

    return (
      hostname === 'localhost'        ||
      hostname === '127.0.0.1'        ||
      hostname === '::1'              ||
      hostname.startsWith('127.')     ||
      hostname.startsWith('192.168.')
    );
  } catch (e) {
    return false;
  }
}

function openFullImage(elem) {
    elemImage = elem.parentElement.parentElement.getElementsByClassName("image-display")[0].children[0];
    src = elemImage.src;
    let new_src;
    if (isLocalServerURL(src)) {
        new_src = src.replace(smallImgView, imgView, 1);
    } else {
        new_src = src;
    }
    window.open(new_src, '_blank');
}

function imageSelector(elem, force=false, recount=true, image=false) {
    if (image) {
        elem = elem.parentElement.getElementsByClassName("image-meta-chance-checkbox")[0];
    }

    if (force) {
        elem.checked = !elem.checked;
    }

    let isChecked = elem.checked;
    let imgContainer = elem.parentElement.parentElement.parentElement;
    if (isChecked) {
        imgContainer.classList.add('image-selected');
    } else {
        imgContainer.classList.remove('image-selected');
    }

    if (recount) {
        updateCheckedImagesCount();
    }
}

function currentSelectedClagnoscoClass() {
    return document.getElementsByClassName("class-selected")[0];
}

function selectClagnoscoClass(elemBtn) {
    const classTabLoading = document.getElementsByClassName("class-tab-loading")[0];
    const classTab = document.getElementById("classTab");
    classTabLoading.classList.remove('hidden');
    classTab.classList.add('hidden');

    let selectedButton = elemBtn;
    let selectedClagnoscoClass = elemBtn.parentElement;
    let currentClagnoscoClass = currentSelectedClagnoscoClass();
    let currentButton;

    if (currentClagnoscoClass !== undefined) {
        currentButton = currentClagnoscoClass.children[0];
        currentButton.disabled = false;
        currentClagnoscoClass.classList.remove('class-selected');
    }

    selectedButton.disabled = true;
    selectedClagnoscoClass.classList.add('class-selected');

    // The rest
    // acceptImageChanges(); sending to server

    let nameTabID = document.getElementsByClassName("class-name-tab-main-index")[0];
    let nameTabName = document.getElementsByClassName("class-name-tab-main-name")[0];
    let nameTabSize = document.getElementsByClassName("class-name-tab-main-size")[0];
    let originName;
    let selectedButtonChildren = selectedButton.children;
    
    nameTabID.textContent = selectedButtonChildren[0].textContent;
    nameTabSize.textContent = selectedButtonChildren[2].textContent;

    originName = selectedButtonChildren[1].textContent;
    if (originName === '') {
        nameTabName.classList.add('name-empty');
    } else {
        nameTabName.classList.remove('name-empty');
    }
    nameTabName.textContent = originName;
    
    const imagesTab = document.getElementById("imagesTab");
    
    imageProbsOrder().then(() => {
        imagesTab.scrollTop = 0;
        classTabLoading.classList.add('hidden');
        classTab.classList.remove('hidden');
    }).catch(error => {
        console.error("Error in imageProbsOrder:", error);
        imagesTab.scrollTop = 0;
        classTabLoading.classList.add('hidden');
        classTab.classList.remove('hidden');
    });
    // imageProbsOrder();

    // setTimeout(() => {
    //     classTabLoading.classList.add('hidden');
    //     classTab.classList.remove('hidden');
    // }, 10);
}

function deselectClagnoscoClass() {
    let currentClagnoscoClass = currentSelectedClagnoscoClass();
    let currentButton;

    if (currentClagnoscoClass !== undefined) {
        currentButton = currentClagnoscoClass.children[0];
        currentButton.disabled = false;
        currentClagnoscoClass.classList.remove('class-selected');
    }
    
    document.getElementById("classTab").classList.add('hidden');
    let classTabLoading = document.getElementsByClassName("class-tab-loading")[0];
    classTabLoading.classList.add('hidden');
}

function populateClagnoscoClasses() {
    deleteAllClagnoscoClasses(confirmDeletingAll=false);

    for (let i = 0; i < clagnoscoClassesSizes.length; i++) {
        baseAddClagnoscoClassTemplate(nameText=clagnoscoClassesSizes[i][0], sizeText=clagnoscoClassesSizes[i][1]);
    }

    document.getElementById("classesList").scrollTop = 0;
    document.getElementsByClassName('classes-tab')[0].classList.remove('hidden');
}

function renameClagnoscoClass() {
    let currentClagnoscoClass = currentSelectedClagnoscoClass();
    if (currentClagnoscoClass === undefined) {
        return null;
    }

    let currentClagnoscoClassText;
    currentClagnoscoClassText = currentClagnoscoClass.children[0].children[1];
    oldName = currentClagnoscoClassText.textContent;

    let newName = prompt("Переименовать класс:", oldName);
    if (newName === null) {
        return null;
    }
    newName = newName.trim();
    let isNewNameEmpty = newName === '';

    currentClagnoscoClassText.textContent = newName;
    if (isNewNameEmpty) {
        currentClagnoscoClassText.classList.add('name-empty');
    } else {
        currentClagnoscoClassText.classList.remove('name-empty');
    }

    let nameTabName = document.getElementsByClassName("class-name-tab-main-name")[0];
    if (isNewNameEmpty) {
        nameTabName.classList.add('name-empty');
    } else {
        nameTabName.classList.remove('name-empty');
    }
    nameTabName.textContent = newName;

    
    // Send changes to server
}

function createEmptyClagnoscoClass(confirmCreatingEmpty=true) {
    if (confirmCreatingEmpty) {
        const confirmStatus = window.confirm(`Создать пустой класс?`);
        if (!confirmStatus) {
            return null;
        }
    }

    addEmptyClagnoscoClassServer();
}

function copyClagnoscoClass(currentButtonCopy=undefined, confirmCopying=true) {
    let currentClagnoscoClass;
    let currentButton;
    if (currentButtonCopy === undefined) {
        currentClagnoscoClass = currentSelectedClagnoscoClass();
    } else {
        currentClagnoscoClass = currentButtonCopy.parentElement;
    }
    currentButton = currentClagnoscoClass.children[0];
    let currentButtonChildren = currentButton.children;
    let nameText = currentButtonChildren[1].textContent;
    let idText = parseInt(currentButtonChildren[0].textContent.replace('№', '', 1)) - 1;
    let newName = nameText + " — копия";
    
    if (confirmCopying) {
        const confirmStatus = window.confirm(`Копировать класс «${nameText}»?`);

        if (!confirmStatus) {
            return null;
        }
    }

    let sizeText = bracketsRemoval(currentButtonChildren[2].textContent);
    copyClagnoscoClassServer(idText, newName, sizeText);

    // acceptImageChanges(); sending to server
    // Send changes to server
}

function redoIndexing() {
    let indexesClagnoscoClasses = document.getElementsByClassName("class-index-name-size-index");
    let indexNameTabElement = document.getElementsByClassName("class-name-tab-main-index")[0];
    let indexNameTab = indexNameTabElement.textContent;
    let oldIndex, newIndex;

    for (let i = 0; i < indexesClagnoscoClasses.length; i++) {
        oldIndex = indexesClagnoscoClasses[i].textContent;
        newIndex = `№${i+1}`;
        indexesClagnoscoClasses[i].textContent = newIndex;
        if (oldIndex == indexNameTab) {
            indexNameTabElement.textContent = newIndex;
        }
    }
}

function deleteClagnoscoClass(currentButtonDelete=undefined, confirmDeleting=true) {
    let currentClagnoscoClass;
    let currentButton;
    let isSelected = false;
    if (currentButtonDelete === undefined) {
        currentClagnoscoClass = currentSelectedClagnoscoClass();
        isSelected = true;
    } else {
        currentClagnoscoClass = currentButtonDelete.parentElement;
        if (currentClagnoscoClass.classList.contains('class-selected')) {
            isSelected = true;
        }
    }
    currentButton = currentClagnoscoClass.children[0];
    let currentButtonChildren = currentButton.children;
    let nameText = currentButtonChildren[1].textContent;
    let idText = parseInt(currentButtonChildren[0].textContent.replace('№', '', 1)) - 1;

    if (confirmDeleting) {
        const confirmStatus = window.confirm(`Удалить класс «${nameText}»?`);

        if (!confirmStatus) {
            return null;
        }
    }

    deleteClagnoscoClassServer(idText, currentClagnoscoClass, isSelected);
}

function deleteAllClagnoscoClasses(confirmDeletingAll=true) {
    if (confirmDeletingAll) {
        const confirmStatus = window.confirm(`Удалить все классы?`);

        if (!confirmStatus) {
            return null;
        }
    }

    deselectClagnoscoClass();
    let classesList = document.getElementById('classesList');
    classesList.innerHTML = '';
    
    // Send changes to server
}

function collectImagesInfo() {
    if (document.getElementById("classTab").classList.contains("hidden")) {
        return null;
    }
    let images = document.getElementsByClassName("image-container");
    var info = [];
    for (let i = 0; i < images.length; i++) {
        imageName = images[i].getElementsByClassName("image-meta-name")[0].textContent;
        imageProb = images[i].getElementsByClassName("image-meta-chance-value")[0].title;
        imagePercent = images[i].getElementsByClassName("image-meta-chance-value")[0].textContent;
        imagePercent = parseFloat(imageProb.split("%", 1)[0]);
        imageChecked = images[i].getElementsByClassName("image-meta-chance-checkbox")[0].checked;
        info.push([images[i], imageName, imagePercent, imageChecked, imageProb]);
    }
    return info;
}

function setPercentImages(newPercent=null) {
    if (document.getElementById("classTab").classList.contains("hidden")) {
        return null;
    }

    let warningText = `При вводе положительного процента верятности попадания в текущий класс будут выбраны только изображения с равным или большим значением верятности.
При вводе отрицательного процента верятности попадания в текущий класс будут выбраны только изображения с равным или меньшим значением верятности.
После ввода значения действие нельзя отменить!`;

    if (newPercent === null) {
        newPercent = prompt(warningText);
        if (newPercent === null) {
            return null;
        }
        newPercent = newPercent.replaceAll(",", ".").replaceAll("%", "");
    }
    newPercent = parseFloat(newPercent) * 0.01;

    let imagesInfo = collectImagesInfo();
    let currentChecked, currentElem, currentProb;

    if (newPercent >= 0) {
        for (let i = 0; i < imagesInfo.length; i++) {
            currentElem = imagesInfo[i][0].getElementsByClassName("image-meta-chance-checkbox")[0];
            currentChecked = currentElem.checked;
            currentProb = parseFloat(imagesInfo[i][4]);
            if (currentProb >= newPercent) {
                if (!currentChecked) {
                    imageSelector(currentElem, force=true, recount=false);
                }
            } else {
                if (currentChecked) {
                    imageSelector(currentElem, force=true, recount=false);
                }
            }
        }
    } else {
        newPercent = -newPercent;
        for (let i = 0; i < imagesInfo.length; i++) {
            currentElem = imagesInfo[i][0].getElementsByClassName("image-meta-chance-checkbox")[0];
            currentChecked = currentElem.checked;
            if (imagesInfo[i][4] <= newPercent) {
                if (!currentChecked) {
                    imageSelector(currentElem, force=true, recount=false);
                }
            } else {
                if (currentChecked) {
                    imageSelector(currentElem, force=true, recount=false);
                }
            }
        }
    }
    updateCheckedImagesCount();
}

function updateCheckedImagesCount() {
    if (document.getElementById("classTab").classList.contains("hidden")) {
        return null;
    }

    let imagesInfo = collectImagesInfo();
    let imagedCheckedCount = 0;
    for (let i = 0; i < imagesInfo.length; i++) {
        currentElem = imagesInfo[i][0].getElementsByClassName("image-meta-chance-checkbox")[0];
        if (currentElem.checked) {
            imagedCheckedCount++;
        }
    }
    let imagedCheckedCountBrackets = `(${imagedCheckedCount})`;

    let currentClagnoscoClass = currentSelectedClagnoscoClass();
    let currentButtonSize;

    if (currentClagnoscoClass !== undefined) {
        currentButtonSize = currentClagnoscoClass.getElementsByClassName("class-index-name-size-size")[0];
        currentButtonSize.textContent = imagedCheckedCountBrackets;
    }

    let nameTabSize = document.getElementsByClassName("class-name-tab-main-size")[0];
    nameTabSize.textContent = imagedCheckedCountBrackets;
}

function selectAllImages(confirmSelectingAllImages=true) {
    if (confirmSelectingAllImages) {
        const confirmStatus = window.confirm(`Выбрать все изображения?`);
        if (!confirmStatus) {
            return null;
        }
    }
    setPercentImages(-101);
}

function deselectAllImages(confirmDeselectingAllImages=true) {
    if (confirmDeselectingAllImages) {
        const confirmStatus = window.confirm(`Внять выделение всем изображениям?`);
        if (!confirmStatus) {
            return null;
        }
    }
    setPercentImages(101);
}


function deleteAllImages() {
    document.getElementsByClassName("images-tab")[0].innerHTML = '';
}

function populateImages(probs=undefined) {
    deleteAllImages();

    if (probs === undefined) {
        for (let i = 0; i < clagnoscoImagesNames.length; i++) {
            baseAddImageContainerTemplate(imageSrc=imgView+imagesFolder+clagnoscoImagesNames[i],
                                          imageName=clagnoscoImagesNames[i]);
        }
    } else {
        probs.forEach(([name, prob, isMember]) => {
            baseAddImageContainerTemplate(imageSrc=imgView+imagesFolder+name,
                                          imageName=name,
                                          prob=prob,
                                          isMember=isMember);
        });
    }
    
    updateCheckedImagesCount();
}

function imageProbsOrder() {
    let currentClagnoscoClass = currentSelectedClagnoscoClass();
    let clagnoscoClassID = parseInt(currentClagnoscoClass.children[0].children[0].textContent.replace('№', '', 1)) - 1;

    let instruction = {'command': 'imageProbsGet',
                       'id': clagnoscoClassID};
    
    return sendToServer(instruction).then(answer => {
        if (answer["status"] === "imagesProbs") {
            populateImages(answer["probs"]);
        } else {
            // alert("Странный ответ сервера.");
        }
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
    });
}


function clearModels() {
    document.getElementById('modelNameSelect').innerHTML = '';
}


function populateModels(localModelsInclude=true, internetModelDownloadInclude=true) {
    if (localModelsInclude || internetModelDownloadInclude) {
        clearModels();
    }

    if (reloadNeeded) {
        if (serverReconnect) {
            baseAddModelTemplate("", "(Соединение восстановлено. Перезагрузите страницу)");
        } else {
            baseAddModelTemplate("", "(Нет соединения с сервером)");
        }
        return null;
    }

    if (internetModelDownloadInclude) {
        baseAddModelTemplate();
    }

    if (localModelsInclude) {
        let instruction = {'command': 'modelsInFolder'};

        sendToServer(instruction).then(answer => {
            if (answer["status"] === "readyToInit") {
                let modelNames = answer["modelNames"];
                for (let i = 0; i < modelNames.length; i++) {
                    baseAddModelTemplate(modelName=modelNames[i]);
                }
                let modelOptions = document.getElementById("modelNameSelect");
                if (modelOptions.children.length > 1) {
                    modelOptions.selectedIndex = 1;
                }

            } else {
                // alert("Странный ответ сервера.");
            }
        }).catch(error => {
            console.error("Ошибка обработки запроса:", error);
        });
    }
}

function addEmptyClagnoscoClassServer() {
    let instruction = {'command': 'addEmptyClagnoscoClass'};
    
    return sendToServer(instruction).then(answer => {
        if (answer["status"] === "emptyClagnoscoClassAdded") {
            baseAddClagnoscoClassTemplate(nameText="Пустой класс", sizeText=0);
        } else if (answer["status"] === "error") {
            alert(answer["message"]);
        } else {
            // alert("Странный ответ сервера.");
        }
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
    });
}

function copyClagnoscoClassServer(clagnoscoClassID, clagnoscoClassName, clagnoscoClassSize) {
    let instruction = {'command': 'copyClagnoscoClass',
                       'id': clagnoscoClassID,
                       'newName': clagnoscoClassName
                      };
    
    return sendToServer(instruction).then(answer => {
        if (answer["status"] === "clagnoscoClassCopied") {
            baseAddClagnoscoClassTemplate(nameText=clagnoscoClassName, sizeText=clagnoscoClassSize);
        } else if (answer["status"] === "error") {
            alert(answer["message"]);
        } else {
            // alert("Странный ответ сервера.");
        }
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
    });
}

function deleteClagnoscoClassServer(clagnoscoClassID, currentClagnoscoClass, isSelected) {
    let instruction = {'command': 'deleteClagnoscoClass',
                       'id': clagnoscoClassID
                      };
    
    return sendToServer(instruction).then(answer => {
        if (answer["status"] === "clagnoscoClassDeleted") {
            if (isSelected) {
                deselectClagnoscoClass();
            }
            currentClagnoscoClass.remove();
            redoIndexing();
        } else if (answer["status"] === "error") {
            alert(answer["message"]);
        } else {
            // alert("Странный ответ сервера.");
        }
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
    });
}


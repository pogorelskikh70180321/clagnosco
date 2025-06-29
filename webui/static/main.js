function setupClagnosco() {
    if (!window.sessionId) {
        window.sessionId = crypto.randomUUID();
    }

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
    
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            closeSaveTab();
        } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
            event.preventDefault();
            openSaveTab();
        }
    });

    resetAll(confirmResetAll=false, populate=true);
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

async function sendToServer(data, isBasic=false, isExit=false) {
    data["sessionID"] = window.sessionId;
    try {
        if (loggingFetch) {
            console.log("Запрос:", data);
        }

        if (oldSession) {
            return null;
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
            if (data["command"] === "currectStatus") {
                return false;
            }
        }
        
        if (answer["status"] === "oldSession") {
            serverConnect = true;
            reloadNeeded = true;
            safeReload = true;
            oldSession = true;
            showCustomAlert("Сессия устарела. Перезагрузите страницу.", timeout=-1);

            throw new Error("Сессия устарела. Перезагрузите страницу.");
        }
        
        return answer;

    } catch (error) {
        if (isExit) {
            return true;
        }
        if (oldSession) {
            return null;
        }
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
                    showCustomAlert("Потеряно соединение с сервером. Процесс не сохранён.", timeout=-1);
                }
            }
        } else {
            if (serverConnect) {
                reloadNeeded = true;
                serverConnect = false;
                serverReconnect = false;
                safeReload = true;
            }
            showCustomAlert("Потеряно соединение с сервером. Процесс не сохранён.", timeout=-1);
        }
        return { error: error.message };
    }
}

function clearCache(confirmClearingCache=true) {
    let cacheButton = document.getElementById("initButtonClearCache");
    let importButton = document.getElementById("initButtonImportData");
    let launchButton = document.getElementById("initButtonProcess");
    let imgDir = document.getElementById("localFolder");
    
    if (confirmClearingCache) {
        const confirmStatus = window.confirm(`Очистить кэш в директории «${imgDir.value}»?`);

        if (!confirmStatus) {
            return null;
        }
    }
    cacheButton.disabled = true;
    importButton.disabled = true;
    launchButton.disabled = true;
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
        cacheButton.disabled = false;
        importButton.disabled = false;
        launchButton.disabled = false;
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
        cacheButton.disabled = false;
        importButton.disabled = false;
        launchButton.disabled = false;
    });
}

function changeClusterAuto() {
    let clusterAuto = document.getElementById("clusterAuto");
    let clusterNumber = document.getElementById("clusterNumber");

    if (!clusterAuto.checked) {
        changeClusterNumberFormat();
        clusterNumber.classList.remove("hidden");
    } else {
        clusterNumber.classList.add("hidden");
    }
}

function changeClusterNumberFormat() {
    let clusterNumber = document.getElementById("clusterNumber");
    let value = Math.round(Number(clusterNumber.value));
    let min = parseInt(clusterNumber.min);
    let max = parseInt(clusterNumber.max);
    if (value < min) {
        value = min;
    } else if (value > max) {
        value = max;
    }
    clusterNumber.value = value;
    return value;
}

function getClusterNumber() {
    if (document.getElementById("clusterAuto").checked) {
        return -1;
    } else {
        return changeClusterNumberFormat();
    }

}

function resetAll(confirmResetAll=true, populate=false) {
    if (reloadNeeded) {
        confirmResetAll=false;
    }

    if (confirmResetAll) {
        const confirmStatus = window.confirm(`Сбросить все классы и начать процесс обработки заново?\n(Модель останется загруженной, если её не менять)`);

        if (!confirmStatus) {
            return null;
        }
    }

    let loading = document.getElementsByClassName('fullscreen-loader')[0];
    loading.classList.remove('hidden');

    document.getElementsByClassName('save-fullscreen')[0].classList.add('hidden');

    let imgDir = document.getElementById("localFolder");
    let modelName = document.getElementById("modelNameSelect");
    let caching = document.getElementById("caching");
    let clusterAuto = document.getElementById("clusterAuto");
    let clusterNumber = document.getElementById("clusterNumber");

    imgDir.disabled = false;
    modelName.disabled = false;
    caching.disabled = false;
    clusterAuto.disabled = false;
    clusterNumber.disabled = false;


    let menuStatusInit = document.getElementById("menuStatusInit");
    let menuStatusProcessing = document.getElementById("menuStatusProcessing");
    let menuStatusDone = document.getElementById("menuStatusDone");
    let menuStatusClear = document.getElementById("menuStatusClear");
    
    menuStatusInit.classList.remove("hidden");
    menuStatusProcessing.classList.add("hidden");
    menuStatusDone.classList.add("hidden");
    menuStatusClear.classList.remove("hidden");

    clagnoscoClassesSizes = [];
    clagnoscoImagesNames = [];
    imagesFolder;

    deleteAllImages();
    deleteAllClagnoscoClasses(false);
    endSession().then(() => {
        populateModels(populate=populate).then(() => {
            document.getElementsByClassName('classes-tab')[0].classList.add('hidden');
            document.getElementById('classTab').classList.add('hidden');

            let classTabLoading = document.getElementsByClassName("class-tab-loading")[0];
            classTabLoading.classList.add('hidden');

            safeReload = true;

            setTimeout(() => {
                loading.classList.add('hidden');
            }, 500);
        });
    });
}


function launchProcessing(confirmLaunchProcessing=true) {
    if (confirmLaunchProcessing) {
        const confirmStatus = window.confirm(`Запустить обработку избражений из выбранной папки в классы?`);
        if (!confirmStatus) {
            return null;
        }
    }

    safeReload = false;

    let imgDir = document.getElementById("localFolder");
    if (imgDir.value.trim() === "") {
        alert("Требуется ввести путь к папке с изображениями");
        return false;
    }
    let modelName = document.getElementById("modelNameSelect");
    let caching = document.getElementById("caching");
    let clusterAuto = document.getElementById("clusterAuto");
    let clusterNumber = document.getElementById("clusterNumber");

    let instruction = {'command': 'launchProcessing',
                       'imgDir': imgDir.value,
                       'modelName': modelName.value,
                       'caching': caching.checked,
                       'clusterNumber': getClusterNumber()};
    imgDir.disabled = true;
    modelName.disabled = true;
    caching.disabled = true;
    clusterAuto.disabled = true;
    clusterNumber.disabled = true;

    let menuStatusInit = document.getElementById("menuStatusInit");
    let menuStatusProcessing = document.getElementById("menuStatusProcessing");
    let menuStatusDone = document.getElementById("menuStatusDone");
    let menuStatusClear = document.getElementById("menuStatusClear");
    
    if (modelName.value == "download" || modelName.value == "download-save") {
        menuStatusProcessing.children[1].textContent = "Загрузка модели из интернета...";
    } else {
        menuStatusProcessing.children[1].textContent = "Загрузка модели...";
    }

    menuStatusInit.classList.add("hidden");
    menuStatusProcessing.classList.remove("hidden");
    menuStatusDone.classList.add("hidden");
    menuStatusClear.classList.add("hidden");

    sendToServer(instruction).then(answer => {
        if (answer["status"] === "readyToCluster") {
            if (modelDownloadSave) {
                populateModels();
            }
            clusterImages();
        } else if (answer["status"] === "error") {
            alert(answer["message"]);
            
            resetAll(confirmResetAll=false);
        } else {
            // alert("Странный ответ сервера.");
            
            resetAll(confirmResetAll=false);
        }
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
        
        resetAll(confirmResetAll=false);
    });
}

function clusterImages(imagesCount=-1, isImport=false) {
    let menuStatusProcessingText = document.getElementById("menuStatusProcessing").children[1];
    let instruction;
    if (!isImport) {
        if (imagesCount === -1) {
            menuStatusProcessingText.textContent = "Обработка изображений...";
        } else {
            menuStatusProcessingText.textContent = `Обработка изображений (${imagesCount})...`;
        }
        instruction = {'command': 'clusterImages'};
    } else {
        instruction = {'command': 'clusterImagesFake'};
    }
    
    sendToServer(instruction).then(answer => {
        if (answer["status"] === "readyToPopulate") {
            // alert("Processing launched successfully.");
            // console.log(answer);
            
            clagnoscoClassesSizes = answer["classesSizes"];
            clagnoscoImagesNames = answer["imagesNames"];
            imagesFolder = answer["imagesFolder"];
            
            menuStatusProcessingText.textContent = "Распределение кластеров...";
            
            if (!isImport) {
                showCustomAlert(`Все изображения обработаны (${clagnoscoImagesNames.length}). Было найдено следующее количество кластеров: ${clagnoscoClassesSizes.length}`);

            } else {
                showCustomAlert(`Таблица была успешно импортирована. Было загружено следующее количество классов: ${clagnoscoClassesSizes.length}`);

            }

            populateClagnoscoClasses();
            // populateImages();

            let menuStatusInit = document.getElementById("menuStatusInit");
            let menuStatusProcessing = document.getElementById("menuStatusProcessing");
            let menuStatusDone = document.getElementById("menuStatusDone");
            let menuStatusClear = document.getElementById("menuStatusClear");

            menuStatusInit.classList.add("hidden");
            menuStatusProcessing.classList.add("hidden");
            menuStatusDone.classList.remove("hidden");
            menuStatusClear.classList.remove("hidden");

        } else if (answer["status"] === "error") {
            alert(answer["message"]);
            
            resetAll(confirmResetAll=false);
        } else {
            // alert("Странный ответ сервера.");
            
            resetAll(confirmResetAll=false);
        }
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
        
        resetAll(confirmResetAll=false);
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
  } catch (error) {
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

function currentSelectedClagnoscoClassID() {
    let selectedClagnoscoClass = currentSelectedClagnoscoClass();
    let idValue;
    if (selectedClagnoscoClass === undefined) {
        idValue = -1;
    } else {
        idValue = parseInt(selectedClagnoscoClass.children[0].children[0].textContent.replace(/[^0-9]/g, '')) - 1;
    }
    return idValue;
}

function selectClagnoscoClass(elemBtn) {
    disableAllClagnoscoClasses();
    clagnoscoClassImagesSelectionUpdate().then(answerUpdate => {
        let classSelectedElement = document.getElementsByClassName('class-selected');
        if (classSelectedElement.length !== 0) {
            classSelectedElement[0].classList.remove('class-selected');
        }
        
        const classTabLoading = document.getElementsByClassName("class-tab-loading")[0];
        classTabLoading.classList.add('hidden');
        const classTab = document.getElementById("classTab");
        classTab.classList.add('hidden');
        classTabLoading.classList.add('hidden');

        elemBtn.parentElement.classList.add('class-selected');
        return true;
    }).then(() => {
        const classTabLoading = document.getElementsByClassName("class-tab-loading")[0];
        const classTab = document.getElementById("classTab");
        classTabLoading.classList.remove('hidden');
        classTab.classList.add('hidden');

        let nameTabID = document.getElementsByClassName("class-name-tab-main-index")[0];
        let nameTabName = document.getElementsByClassName("class-name-tab-main-name")[0];
        let nameTabSize = document.getElementsByClassName("class-name-tab-main-size")[0];

        let selectedButton = elemBtn;
        let selectedButtonChildren = selectedButton.children;
        
        nameTabID.textContent = selectedButtonChildren[0].textContent;
        nameTabSize.textContent = selectedButtonChildren[2].textContent;

        let originName = selectedButtonChildren[1].textContent;
        if (originName === '') {
            nameTabName.classList.add('name-empty');
        } else {
            nameTabName.classList.remove('name-empty');
        }
        nameTabName.textContent = originName;

        let idText = parseInt(nameTabID.textContent.replace(/[^0-9]/g, '')) - 1;
        imageProbsOrder(idText).then(() => {
            classTabLoading.classList.add('hidden');
            classTab.classList.remove('hidden');
            enableAllClagnoscoClasses();
        }).catch(error => {
            console.error("Error in imageProbsOrder:", error);
            elemBtn.parentElement.classList.add('class-selected');

            classTabLoading.classList.add('hidden');
            classTab.classList.remove('hidden');
            enableAllClagnoscoClasses();
        });
    });
}

function deselectClagnoscoClass(update=true) {
    disableAllClagnoscoClasses();
    clagnoscoClassImagesSelectionUpdate(update=update).then(answerUpdate => {
        let classSelectedElement = document.getElementsByClassName('class-selected');
        if (classSelectedElement.length !== 0) {
            classSelectedElement[0].classList.remove('class-selected');
        }

        document.getElementsByClassName("class-name-tab-main-index")[0].textContent = '';
        document.getElementsByClassName("class-name-tab-main-name")[0].textContent = '';
        document.getElementsByClassName("class-name-tab-main-size")[0].textContent = '';

        if (answerUpdate === -1) {
            document.getElementById("classTab").classList.add('hidden');
        }
        
        const classTabLoading = document.getElementsByClassName("class-tab-loading")[0];
        classTabLoading.classList.add('hidden');
        const classTab = document.getElementById("classTab");
        classTab.classList.add('hidden');
        classTabLoading.classList.add('hidden');
        enableAllClagnoscoClasses();
        return true;
    });
}

function disableAllClagnoscoClasses() {
    document.getElementsByClassName("class-button-add")[0].disabled = true;
    document.getElementsByClassName("class-button-class-not-element")[0].disabled = true;

    let clagnoscoClassButtons = document.getElementsByClassName("class-index-name-size");
    let copyButtons = document.getElementsByClassName("class-button-copy");
    let deleteButtons = document.getElementsByClassName("class-button-delete");
    for (let i = 0; i < clagnoscoClassButtons.length; i++) {
        clagnoscoClassButtons[i].disabled = true;
        copyButtons[i].disabled = true;
        deleteButtons[i].disabled = true;
    }
}

function enableAllClagnoscoClasses() {
    document.getElementsByClassName("class-button-add")[0].disabled = false;
    document.getElementsByClassName("class-button-class-not-element")[0].disabled = false;

    let clagnoscoClassButtons = document.getElementsByClassName("class-index-name-size");
    let copyButtons = document.getElementsByClassName("class-button-copy");
    let deleteButtons = document.getElementsByClassName("class-button-delete");
    for (let i = 0; i < clagnoscoClassButtons.length; i++) {
        if (!clagnoscoClassButtons[i].parentElement.classList.contains("class-selected")) {
            clagnoscoClassButtons[i].disabled = false;
        }
        copyButtons[i].disabled = false;
        deleteButtons[i].disabled = false;
    }
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
        return -1;
    }

    let currentClagnoscoClassText;
    currentClagnoscoClassText = currentClagnoscoClass.children[0].children[1];
    oldName = currentClagnoscoClassText.textContent;

    let newName = prompt("Переименовать класс:", oldName);
    if (newName === null) {
        return null;
    }
    
    let instruction = {'command': 'renameClagnoscoClass',
                       'id': currentSelectedClagnoscoClassID(),
                       'newName': newName,
                      };
    return sendToServer(instruction).then(answer => {
        if (answer["status"] === "clagnoscoClassRenamed") {
            newName = answer["newName"];

            let isNewNameEmpty = newName == '';

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
        } else if (answer["status"] === "error") {
            alert(answer["message"]);
        } else {
            // alert("Странный ответ сервера.");
        }
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
    });
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

function createRestClagnoscoClass(confirmCreatingRest=true) {
    if (confirmCreatingRest) {
        const confirmStatus = window.confirm(`Создать новый класс из элементов, не входящих ни в один класс?`);
        if (!confirmStatus) {
            return null;
        }
    }
    
    clagnoscoClassImagesSelectionUpdate().then(() => {
        let instruction = {'command': 'createRestClagnoscoClass'};
            
        return sendToServer(instruction).then(answer => {
            if (answer["status"] === "restClagnoscoClassCreated") {
                baseAddClagnoscoClassTemplate(nameText=answer["name"], sizeText=answer["size"]);
            } else if (answer["status"] === "error") {
                alert(answer["message"]);
            } else {
                // alert("Странный ответ сервера.");
            }
            canBeSaved();
        }).catch(error => {
            console.error("Ошибка обработки запроса:", error);
        });
    });
}


function copyClagnoscoClass(currentButtonCopy=undefined, confirmCopying=true) {
    let currentClagnoscoClass;
    let currentButton;
    if (currentButtonCopy === undefined) {
        currentClagnoscoClass = currentSelectedClagnoscoClass();
    } else {
        currentClagnoscoClass = currentButtonCopy.parentElement;
    }

    clagnoscoClassImagesSelectionUpdate().then(() => {
        currentButton = currentClagnoscoClass.children[0];
        let currentButtonChildren = currentButton.children;
        let nameText = currentButtonChildren[1].textContent;
        let idText = parseInt(currentButtonChildren[0].textContent.replace(/[^0-9]/g, '')) - 1;
        let newName = nameText + " — копия";
        
        if (confirmCopying) {
            const confirmStatus = window.confirm(`Копировать класс «${nameText}»?`);

            if (!confirmStatus) {
                return null;
            }
        }
        
        disableAllClagnoscoClasses();

        let sizeText = bracketsRemoval(currentButtonChildren[2].textContent);
        copyClagnoscoClassServer(idText, newName, sizeText);

        // acceptImageChanges(); sending to server
        // Send changes to server
    });
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
    let idText = parseInt(currentButtonChildren[0].textContent.replace(/[^0-9]/g, '')) - 1;

    let saveBtn = document.getElementById("initButtonSave");
    saveBtn.disabled = true;

    if (confirmDeleting) {
        const confirmStatus = window.confirm(`Удалить класс «${nameText}»?`);

        if (!confirmStatus) {
            saveBtn.disabled = false;
            return null;
        }
    }

    disableAllClagnoscoClasses();
    deleteClagnoscoClassServer(idText, currentClagnoscoClass, isSelected);
}

function deleteAllClagnoscoClasses(confirmDeletingAll=true) {
    if (confirmDeletingAll) {
        const confirmStatus = window.confirm(`Удалить все классы?`);

        if (!confirmStatus) {
            return null;
        }
    }

    deselectClagnoscoClass(update=false);
    let classesList = document.getElementById('classesList');
    classesList.innerHTML = '';
    
    document.querySelector(".classes-header").children[0].textContent =
        'Классы (' + document.querySelectorAll('.class-selection').length + '):';

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

function inverseImages(confirmInverseImages=true) {
    if (confirmInverseImages) {
        const confirmStatus = window.confirm(`Инвертировать выделение всем изображениям?`);
        if (!confirmStatus) {
            return null;
        }
    }
    
    let imagesInfo = collectImagesInfo();
    for (let i = 0; i < imagesInfo.length; i++) {
        imageSelector(imagesInfo[i][0].getElementsByClassName("image-meta-chance-checkbox")[0],
                      force=true, recount=false);
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

    let neededSlash = "/";
    if (imagesFolder[imagesFolder.length-1] === "/" || imagesFolder[imagesFolder.length-1] === "\\") {
        neededSlash = "";
    }

    if (probs === undefined) {
        for (let i = 0; i < clagnoscoImagesNames.length; i++) {
            baseAddImageContainerTemplate(imageSrc=smallImgView+imagesFolder+neededSlash+clagnoscoImagesNames[i],
                                          imageName=clagnoscoImagesNames[i]);
        }
    } else {
        probs.forEach(([name, prob, isMember]) => {
            baseAddImageContainerTemplate(imageSrc=smallImgView+imagesFolder+neededSlash+name,
                                          imageName=name,
                                          prob=prob,
                                          isMember=isMember);
        });
    }
    
    updateCheckedImagesCount();
}

function imageProbsOrder(clagnoscoClassID=undefined) {
    if (clagnoscoClassID === undefined) {
        clagnoscoClassID = currentSelectedClagnoscoClassID();
    }

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


async function populateModels(populate=true, localModelsInclude=true, internetModelDownloadInclude=true) {
    if (!populate) {
        return false;
    }

    if (localModelsInclude || internetModelDownloadInclude) {
        clearModels();
    }
    localModelDownloadSave = modelDownloadSave;
    if (reloadNeeded) {
        if (serverReconnect) {
            baseAddModelTemplate("", "(Соединение восстановлено. Перезагрузите страницу)");
        } else {
            baseAddModelTemplate("", "(Нет соединения с сервером)");
        }
        return null;
    }

    let selectIndex = 0;
    if (internetModelDownloadInclude) {
        baseAddModelTemplate();
        selectIndex += 1;
    }

    if (localModelsInclude) {
        let instruction = {'command': 'modelsInFolder'};

        sendToServer(instruction).then(answer => {
            if (answer["status"] === "readyToInit") {
                document.getElementById("projectVersion").textContent = `v${answer["projectVersion"]}`;

                let modelNames = answer["modelNames"];
                let modelDownloaded = false;
                if (internetModelDownloadInclude) {
                    for (let i = 0; i < modelNames.length; i++) {
                        if (modelNames[i] == "model.pt") {
                            modelDownloaded = true;
                        }
                    }

                    if (!modelDownloaded) {
                        baseAddModelTemplate("download-save", "model.pt (загрузка с сохранением на диск)");
                        selectIndex += 1;
                        modelDownloadSave = true;
                    }
                }

                for (let i = 0; i < modelNames.length; i++) {
                    baseAddModelTemplate(modelName=modelNames[i]);
                }
                let modelOptions = document.getElementById("modelNameSelect");

                if (localModelDownloadSave) {
                    for (let modelOption = 0; modelOption < modelOptions.children.length; modelOption++) {
                        if (modelOptions.children[modelOption].textContent == "model.pt") {
                            selectIndex = modelOption;
                            modelDownloadSave = false;
                        }
                    }
                }

                if (modelOptions.children.length > selectIndex) {
                    modelOptions.selectedIndex = selectIndex;
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
        canBeSaved();
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
            enableAllClagnoscoClasses();
        } else if (answer["status"] === "error") {
            alert(answer["message"]);
            enableAllClagnoscoClasses();
        } else {
            // alert("Странный ответ сервера.");
        }
        canBeSaved();
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
        enableAllClagnoscoClasses();
        canBeSaved();
    });
}

function deleteClagnoscoClassServer(clagnoscoClassID, currentClagnoscoClass, isSelected) {
    let instruction = {'command': 'deleteClagnoscoClass',
                       'id': clagnoscoClassID
                      };
    let classesAmount;
    return sendToServer(instruction).then(answer => {
        if (answer["status"] === "clagnoscoClassDeleted") {
            if (isSelected) {
                deselectClagnoscoClass(update=false);
            }
            currentClagnoscoClass.remove();
            redoIndexing();
            enableAllClagnoscoClasses();
            
            classesAmount = document.querySelectorAll('.class-selection').length;
            document.querySelector(".classes-header").children[0].textContent =
                'Классы (' + classesAmount + '):';
            

        } else if (answer["status"] === "error") {
            alert(answer["message"]);
            enableAllClagnoscoClasses();
        } else {
            // alert("Странный ответ сервера.");
        }
        canBeSaved();
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
        enableAllClagnoscoClasses();
        canBeSaved();
    });
}

function canBeSaved() {
    let saveBtn = document.getElementById("initButtonSave");
    let classesAmount = document.querySelectorAll('.class-selection').length;
    if (classesAmount === 0) {
        saveBtn.disabled = true;
        saveBtn.title = "Требуется как минимум один класс";
        saveBtn.classList.add("save-btn-disabled");
    } else {
        saveBtn.disabled = false;
        saveBtn.title = "";
        saveBtn.classList.remove("save-btn-disabled");
    }
}

async function clagnoscoClassImagesSelectionUpdate(update=true) {
    const imagesInfo = collectImagesInfo();
    let imagesSelected = [];
    let nameTabID = currentSelectedClagnoscoClassID();

    if (nameTabID === -1) {
        return -1;
    }

    for (let i = 0; i < imagesInfo.length; i++) {
        imagesSelected.push([imagesInfo[i][1], imagesInfo[i][3]]);
    }
    if (!update) {
        return true;
    }
    let instruction = {'command': 'clagnoscoClassImagesSelectionUpdate',
                       'id': nameTabID,
                       'selection': imagesSelected,
                      };
    
    return sendToServer(instruction).then(answer => {
        if (answer["status"] === "clagnoscoClassImagesSelectionUpdated") {
            
        } else if (answer["status"] === "error") {
            alert(answer["message"]);
        } else {
            // alert("Странный ответ сервера.");
        }
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
    });
}

function openSaveTab(force=false) {
    if (!document.getElementById("menuStatusDone").classList.contains("hidden") || force) {
        clagnoscoClassImagesSelectionUpdate().then(answerUpdate => {
            document.getElementsByClassName("save-fullscreen")[0].classList.remove("hidden");
            document.querySelector('.container').setAttribute('inert', '');
            document.querySelector(".save-tab").querySelector(".close-button").focus();
        });
    }
}

function closeSaveTab() {
    if (!isSaving) {
        let fullScreenElem = document.getElementsByClassName("save-fullscreen")[0];
        if (!fullScreenElem.classList.contains("hidden")) {
            fullScreenElem.classList.add("hidden");
            document.querySelector('.container').removeAttribute('inert');
            document.getElementById('initButtonSave').focus();
        }
    }
}

function saveFolder(confirmSaveFolder=true) {
    if (isSaving) {
        return false;
    }

    let saveFolderType = document.querySelector('input[name="saveFolderType"]:checked').value;

    let warningText = "Сохранить классы в субдиректориях?";
    if (saveFolderType === "numberName") {
        warningText = `Сохранить классы в субдиректориях как имя и номер класса?\n(Названия папок сократятся до 100 символов)`;
    } else if (saveFolderType === "number") {
        warningText = `Сохранить классы в субдиректориях как номер класса?`;
    } else if (saveFolderType === "name") {
        warningText = `Сохранить классы в субдиректориях как имя класса?\n(Названия папок сократятся до 100 символов)`;
    }

    if (confirmSaveFolder) {
        const confirmStatus = window.confirm(warningText);
        if (!confirmStatus) {
            return null;
        }
    }

    isSaving = true;
    savingStatusButtons(true, "folder");

    let instruction = {'command': 'saveFolder',
                       'namingType': saveFolderType
                      };
    
    sendToServer(instruction).then(answer => {
        if (answer["status"] === "saveFolderSuccess") {
            showCustomAlert(`Все изображения успешно распределены по субдиректириям в папке «${answer["folder"]}»`);
        } else if (answer["status"] === "error") {
            console.log(answer["console"]);
            alert(answer["message"]);
        } else {
            // alert("Странный ответ сервера.");
        }
        isSaving = false;
        savingStatusButtons(false);
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
        isSaving = false;
        savingStatusButtons(false);
    });
}

function saveTable(confirmSaveTable=true) {
    if (isSaving) {
        return false;
    }

    if (confirmSaveTable) {
        const confirmStatus = window.confirm(`Сохранить классы в качестве CSV таблицы?`);
        if (!confirmStatus) {
            return null;
        }
    }

    // savingStatusButtons(true, "table");

    let instruction = {'command': 'saveTable'};
    
    sendToServer(instruction).then(answer => {
        if (answer["status"] === "saveTableSuccess") {
            downloadTextFile(answer["table"], answer["fileName"], "text/csv");
            showCustomAlert('Файл таблицы добавлен в загрузки браузера: ' + answer["fileName"])
        } else if (answer["status"] === "error") {
            alert(answer["message"]);
        } else {
            // alert("Странный ответ сервера.");
        }
        isSaving = false;
        // savingStatusButtons(false);
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
        isSaving = false;
        // savingStatusButtons(false);
    });
}

function downloadTextFile(content, filename, mimeType='text/plain') {
    const blob = new Blob([content], { type: mimeType + ';charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    URL.revokeObjectURL(url);
}

function savingStatusButtons(toSave=false, savingType) {
    let btnTable = document.querySelector(".save-btn-table");
    let btnFolder = document.querySelector(".save-btn-folder");
    let radioInputs = document.querySelector(".radio-tab").querySelectorAll('input[type="radio"]');

    if (!toSave) {
        btnTable.classList.remove("save-btn-loading");
        btnTable.classList.remove("save-btn-disabled");
        btnTable.textContent = "Сохранить в качестве CSV таблицы";
        btnFolder.classList.remove("save-btn-loading");
        btnFolder.classList.remove("save-btn-disabled");
        btnFolder.textContent = "Сохранить в субдиректориях";
        for (let i = 0; i < radioInputs.length; i++) {
            radioInputs[i].disabled = false;
            radioInputs[i].classList.remove("radio-disabled");
        }
        return null;
    }

    for (let i = 0; i < radioInputs.length; i++) {
        radioInputs[i].disabled = true;
        radioInputs[i].classList.add("radio-disabled");
    }

    if (savingType === "table") {
        btnTable.classList.add("save-btn-loading");
        btnTable.classList.remove("save-btn-disabled");
        btnTable.textContent = "Сохранение в качестве CSV таблицы...";

        btnFolder.classList.remove("save-btn-loading");
        btnFolder.classList.add("save-btn-disabled");
    } else if (savingType === "folder") {
        btnTable.classList.remove("save-btn-loading");
        btnTable.classList.add("save-btn-disabled");

        btnFolder.classList.add("save-btn-loading");
        btnFolder.classList.remove("save-btn-disabled");
        btnFolder.textContent = "Сохранение в субдиректориях...";
    }
}

function unloadModel(confirmUnloadingModel=true) {
    if (confirmUnloadingModel) {
        const confirmStatus = window.confirm(`Выгрузить модель из памяти?\n(Модель не будет удалена с устройства)`);
        if (!confirmStatus) {
            return null;
        }
    }

    let instruction = {'command': 'unloadModel'};
    
    sendToServer(instruction).then(answer => {
        if (answer["status"] === "modelUnloaded") {
            showCustomAlert(answer["message"]);
        } else if (answer["status"] === "error") {
            alert(answer["message"]);
        } else {
            // alert("Странный ответ сервера.");
        }
        isSaving = false;
    }).catch(error => {
        console.error("Ошибка обработки запроса:", error);
        isSaving = false;
    });
}

async function endSession() {
    let instruction = {'command': 'endSession'};
    
    sendToServer(instruction).then(answer => {
    }).catch(error => {
    });
}

function importData() {
    const fileInput = document.getElementById('csvFileInput');
    const file = fileInput.files[0];

    if (!file) {
        console.log("Файл импорта не выбран");
        return;
    }

    const fileName = file.name.toLowerCase();
    if (!fileName.endsWith('.csv')) {
        console.log("Неверный формат импорта файла. Требуется .csv");
        fileInput.value = "";
        return;
    }

    let imgDir = document.getElementById("localFolder");
    let modelName = document.getElementById("modelNameSelect");
    let caching = document.getElementById("caching");
    let clusterAuto = document.getElementById("clusterAuto");
    let clusterNumber = document.getElementById("clusterNumber");

    imgDir.disabled = true;
    modelName.disabled = true;  
    caching.disabled = true;
    clusterAuto.disabled = true;
    clusterNumber.disabled = true;

    
    let menuStatusProcessingText = document.getElementById("menuStatusProcessing").children[1];
    menuStatusProcessingText.textContent = "Импорт данных...";
    
    let menuStatusInit = document.getElementById("menuStatusInit");
    let menuStatusProcessing = document.getElementById("menuStatusProcessing");
    let menuStatusDone = document.getElementById("menuStatusDone");
    let menuStatusClear = document.getElementById("menuStatusClear");
    
    menuStatusProcessing.children[1].textContent = "Импорт данных...";

    menuStatusInit.classList.add("hidden");
    menuStatusProcessing.classList.remove("hidden");
    menuStatusDone.classList.add("hidden");
    menuStatusClear.classList.add("hidden");

    
    const reader = new FileReader();

    reader.onload = function (e) {
        const csvText = e.target.result;
        
        const instruction = {
            command: 'importData',
            table: csvText
        };

        sendToServer(instruction).then(answer => {
            if (answer["status"] === "dataImported") {
                imgDir.value = answer["imgDir"];
                clusterImages(-1, true);
            } else if (answer["status"] === "error") {
                alert(answer["message"]);
                resetAll(confirmResetAll=false);
            } else {
                resetAll(confirmResetAll=false);
            }
            fileInput.value = "";
        }).catch(error => {
            console.error("Ошибка обработки запроса:", error);
            resetAll(confirmResetAll=false);
            fileInput.value = "";
        });
    };

    reader.onerror = function () {
        console.error("Ошибка чтения CSV-файла.");
        fileInput.value = "";
    };

    reader.readAsText(file, "utf-8");
}

function currentStatus() {
    let instruction = {'command': 'currentStatus'};

    sendToServer(instruction, false, true).then().catch(error => {
        alert("Ошибка при получении состояния");
    });
}

async function exitClagnosco(confirmExitClagnosco=true) {
    if (confirmExitClagnosco) {
        const confirmStatus = window.confirm(`Вы действительно хотите завершить работу Clagnosco?`);
        if (!confirmStatus) {
            return null;
        }
    }
    
    if (!serverConnect) {
        showCustomAlert("Работа Clagnosco уже была завершена", -1)
        safeReload = true;
        return false;
    }
    
    let instruction = {'command': 'exitClagnosco'};

    sendToServer(instruction, false, true).then(answer => {
        if (answer) {
            showCustomAlert("Работа Clagnosco завершена", -1)
            console.log("Работа Clagnosco завершена");
            serverConnect = false;
            safeReload = true;
        } else {
            console.log("Странное выполнение завершения Clagnosco");
        }
    }).catch(error => {
        alert("Ошибка при завершении программы");
    });
}

function printProcess(manual=undefined) {
    let instruction = {'command': 'printProcess'};
    if (manual !== undefined) {
        instruction['printProcess'] = manual;
    }
    
    return sendToServer(instruction).then().catch(error => {
        console.error("Ошибка обработки запроса:", error);
    });
}

function info() {
    console.log("currentStatus();");
    console.log("printProcess(manual=undefined); // manual = true для печати процесса");
}

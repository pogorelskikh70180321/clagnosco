var isProcessing = false;


function sendToServer(instruction="") {


    // var fullInstruction = {command: ''};
    // var sendingType = "give";
    // if (instruction === "") {
    //     console.log("Попытка отправки пустой инструкции.");
    //     return null;
    
    // } else if (instruction === "launchProceccing") {
    //     fullInstruction['command'] = 'launchProceccing';
    //     fullInstruction['clagnoscoClassID'] = extra['clagnoscoClassID'];

    // } else if (instruction === "getClagnoscoClasses") {
    //     fullInstruction['command'] = 'getClagnoscoClasses';
    //     sendingType = 'receive';

    // } else if (instruction === "getImages") {
    //     fullInstruction['command'] = 'getImages';
    //     sendingType = 'receive';

    // } else if (instruction === "creatingEmptyClagnoscoClass") {
    //     fullInstruction['command'] = 'creatingEmptyClagnoscoClass';
    
    // } else if (instruction === "copyingClagnoscoClass") {
    //     fullInstruction['command'] = 'copyingClagnoscoClass';
    //     fullInstruction['clagnoscoClassID'] = extra['clagnoscoClassID'];
    // }
}

function launchProceccing() {
    folder = document.getElementById("localFolder").textContent;
    model = document.getElementById("modelSelect").value;
    cashing = document.getElementById("cashing").checked;
    instruction = {'command': 'launchProceccing', 'folder': folder, 'model': model, 'cashing': cashing};
    isProcessing = true;
    sendToServer(instruction);
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
        new_src = src.replace('/img_small/', '/img/', 1);
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

    document.getElementById("classTab").classList.remove('hidden');

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



    // repopulateProbs(id); receiving from server
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
    baseAddClagnoscoClassTemplate(nameText="Пустой класс", sizeText=0);

    // Send changes to server
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
    
    if (confirmCopying) {
        const confirmStatus = window.confirm(`Копировать класс «${nameText}»?`);

        if (!confirmStatus) {
            return null;
        }
    }

    let sizeText = bracketsRemoval(currentButtonChildren[2].textContent);
    baseAddClagnoscoClassTemplate(nameText=nameText + " — копия", sizeText=sizeText);

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
    
    if (confirmDeleting) {
        const confirmStatus = window.confirm(`Удалить класс «${nameText}»?`);

        if (!confirmStatus) {
            return null;
        }
    }
    if (isSelected) {
        deselectClagnoscoClass();
    }
    currentClagnoscoClass.remove();


    redoIndexing();
    // Send changes to server
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

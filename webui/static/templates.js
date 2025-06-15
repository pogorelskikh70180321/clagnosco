const pixelBase64 = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdj+Pf/338ACfUD+5LbVnEAAAAASUVORK5CYII=';

function clagnoscoClassTemplate(indexText=-1, nameText="", sizeText=0) {
    const div = document.createElement('div');
    div.className = 'class-selection';

    div.innerHTML = `
        <button class="class-index-name-size" onclick="selectClagnoscoClass(this);">
            <div class="class-index-name-size-index"></div>
            <div class="class-index-name-size-name"></div>
            <div class="class-index-name-size-size"></div>
        </button>
        <button class="class-button class-button-copy" title="Копировать класс" onclick="copyClagnoscoClass(this);"></button>
        <button class="class-button class-button-delete" title="Удалить класс" onclick="deleteClagnoscoClass(this);"></button>
    `;

    const indexDiv = div.querySelector('.class-index-name-size-index');
    const nameDiv = div.querySelector('.class-index-name-size-name');
    const sizeDiv = div.querySelector('.class-index-name-size-size');

    if (indexText == -1) {
        indexesElems = document.getElementsByClassName('class-index-name-size-index');
        indexesElemsLength = indexesElems.length;
        if (indexesElemsLength != 0) {
            indexText = parseInt(indexesElems[indexesElemsLength - 1].textContent.split('№')[1]) + 1;
        } else {
            indexText = 1;
        }
    }
    indexDiv.textContent = '№' + indexText;
    nameDiv.textContent = nameText;
    sizeDiv.textContent = '(' + sizeText + ')';

    if (nameText === "") {
        nameDiv.classList.add('name-empty');
    } else {
        nameDiv.classList.remove('name-empty');
    }

    return div;
}

function baseAddClagnoscoClassTemplate(nameText="", sizeText=0) {
    let div = clagnoscoClassTemplate(indexText=-1, nameText=nameText, sizeText=sizeText);
    let classesListElement = document.getElementById('classesList');
    classesListElement.appendChild(div);

    document.querySelector(".classes-header").children[0].textContent =
        'Классы (' + document.querySelectorAll('.class-selection').length + '):';
}


function imageContainerTemplate(imageSrc=undefined, imageName="Example", prob=0.0, isMember=false) {
    if (imageSrc === undefined) {
        imageSrc = pixelBase64;
    }

    const div = document.createElement('div');
    div.className = 'image-container';

    div.innerHTML = `
                        <div class="image-display" onclick="imageSelector(this, force=true, recount=true, image=true);">
                            <img draggable="false" src="">
                        </div>
                        <div class="image-meta">
                            <div class="image-meta-name" onclick="openFullImage(this);" title='Открыть «» в новой вкладке'></div>
                            <div class="image-meta-chance-tab">
                                <div>Вероятность:</div>
                                <div class="image-meta-chance-value" title='0.0'>0.00%</div>
                                <div>Выбрать:</div>
                                <input type="checkbox" class="image-meta-chance-checkbox" onclick="imageSelector(this);"/>
                            </div>
                        </div>
    `;

    const imageDiv = div.querySelector('.image-display').children[0];
    imageDiv.src = imageSrc;

    const metaNameDiv = div.querySelector('.image-meta-name');
    metaNameDiv.textContent = imageName;
    metaNameDiv.title = `Открыть «${imageName}» в новой вкладке`;

    const probDiv = div.querySelector('.image-meta-chance-value');
    probDiv.title = prob;
    probDiv.textContent = (prob * 100).toFixed(2) + '%';

    if (isMember) {
        div.querySelector('.image-meta-chance-checkbox').checked = true;
        div.classList.add('image-selected');
    }

    return div;
}

function baseAddImageContainerTemplate(imageSrc=undefined, imageName="Example", prob=0.0, isMember=false) {
    let div = imageContainerTemplate(imageSrc=imageSrc, imageName=imageName, prob=prob, isMember=isMember);
    let classesListElement = document.getElementById('imagesTab');
    classesListElement.appendChild(div);
}

function modelTemplate(modelName="download", modelCaption=undefined) {
    const option = document.createElement('option');
    if (modelName === "download") {
        option.textContent = 'model.pt (использование модели из интернета без сохранения на диск)';
    } else {
        option.textContent = modelName;
    }

    if (modelCaption !== undefined) {
        option.textContent = modelCaption;
    }
    option.value = modelName;
    return option
}

function baseAddModelTemplate(modelName="download", modelCaption=undefined) {
    let option;
    if (modelName === undefined) {
        option = modelTemplate(modelName="model.pt", modelCaption=modelCaption);
    } else {
        option = modelTemplate(modelName=modelName, modelCaption=modelCaption);
    }
    let modelsListElement = document.getElementById('modelNameSelect');
    modelsListElement.appendChild(option);
}

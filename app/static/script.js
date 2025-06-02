// Функція для переміщення сервоприводів
function move(direction) {
    fetch('/move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({ direction: direction })
    })
    .then(response => response.text())
    .then(data => console.log(data)) // Виводимо відповідь від сервера (якщо потрібно)
    .catch(error => console.error('Error:', error)); // Обробка помилок
}

// Функція для оновлення списку розпізнаних об'єктів
function updateObjects() {
    fetch('/objects')
        .then(response => response.json())
        .then(data => {
            const objectList = document.getElementById('object-list');
            objectList.innerHTML = ''; // Очищаємо список перед оновленням

            data.forEach((obj, index) => {
                const li = document.createElement('li');
                li.textContent = obj.name; // Використовуємо 'name', щоб відобразити правильну назву

                // Створюємо кнопки для стеження
                const trackButton = document.createElement('button');
                trackButton.textContent = 'Стежити';
                trackButton.onclick = function() {
                    trackObject(index);
                };

                // Створюємо кнопку для припинення стеження
                const stopTrackingButton = document.createElement('button');
                stopTrackingButton.textContent = 'Припинити';
                stopTrackingButton.onclick = function() {
                    stopTrackingObject(index);
                };

                li.appendChild(trackButton);
                li.appendChild(stopTrackingButton);
                objectList.appendChild(li);
            });
        })
        .catch(error => console.error('Error:', error)); // Обробка помилок
}

// Логіка для стеження за об'єктами
function trackObject(index) {
    console.log('Стежимо за об\'єктом з індексом: ', index);
    
    // Надсилаємо ID об'єкта для початку стеження
    fetch('/track', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({ object_id: index })
    })
    .then(response => response.text())
    .then(data => console.log(data)) // Виводимо відповідь від сервера
    .catch(error => console.error('Error:', error)); // Обробка помилок
}

// Логіка для припинення стеження за об'єктами
function stopTrackingObject(index) {
    console.log('Припиняємо стеження за об\'єктом з індексом: ', index);
    
    // Надсилаємо запит на припинення стеження
    fetch('/stop_tracking', {
        method: 'POST'
    })
    .then(response => response.text())
    .then(data => console.log(data)) // Виводимо відповідь від сервера
    .catch(error => console.error('Error:', error)); // Обробка помилок
}

// Оновлюємо список об'єктів кожні 3 секунди
setInterval(updateObjects, 3000);


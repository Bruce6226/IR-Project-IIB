document.getElementById('uploadForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];

    if (!file) {
        alert('Selecciona un archivo.');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    // mostrar la imagen consultada
    const queryImageUrl = URL.createObjectURL(file);
    document.getElementById('queryImage').src = queryImageUrl;

    fetch('http://127.0.0.1:5000/api/v1/search', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            const gallery = document.getElementById('resultsGallery');
            gallery.innerHTML = '';  // limpiar galeria de resultados previos

            document.getElementById('titulo').classList.add('text-dark');
            document.getElementById('tituloR').classList.add('text-dark');

            data.forEach(result => {
                const img = document.createElement('img');
                img.src = `http://127.0.0.1:5000${result.image_url}`;
                img.alt = `Etiqueta: ${result.label} - Distancia: ${result.distance.toFixed(2)}`;
                img.classList.add('result-image');  // añadir clase para estilos
                gallery.appendChild(img);
            });
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Ocurrió un error mientras se buscaban imágenes similares.');
        });
});

const inputFile = document.getElementById('inputFile');
const buttonAnalyze = document.getElementById('buttonAnalyze');
const img = document.getElementById('imagePreview');
const promptContainer = document.getElementById('promptContainer');
const loadingOverlay = document.getElementById('loadingOverlay');

let previewUrl = null;

inputFile.addEventListener('change', () => {
    const file = inputFile.files[0];
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    promptContainer.textContent = '';
    buttonAnalyze.disabled = !file;
    if (!file) {
        img.style.display = 'none';
        return;
    }
    previewUrl = URL.createObjectURL(file);
    img.src = previewUrl;
    img.style.display = 'block';
});

buttonAnalyze.addEventListener('click', async () => {
    const file = inputFile.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);
    loadingOverlay.style.display = 'flex';
    buttonAnalyze.disabled = true;

    try {
        const response = await fetch('/analyze-image', { method: 'POST', body: formData });
        const data = await response.json();
        promptContainer.textContent = response.ok
            ? `Descrição da imagem: ${data.description}`
            : data.error ?? 'Erro ao analisar a imagem.';
    } catch {
        promptContainer.textContent = 'Não foi possível conectar ao servidor.';
    } finally {
        loadingOverlay.style.display = 'none';
        buttonAnalyze.disabled = false;
    }
});

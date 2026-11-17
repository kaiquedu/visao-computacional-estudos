# Descrição de Imagens

Aplicação web que recebe uma imagem e devolve uma descrição em português. O backend em Flask gera uma legenda em inglês com o BLIP e traduz com um modelo MarianMT. Os dois modelos rodam localmente, sem depender de API externa.

## Como funciona

1. O frontend envia a imagem para `POST /analyze-image`.
2. O backend confere se o arquivo é uma imagem válida, com no máximo 5 MB e 16 megapixels.
3. O [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) gera a legenda em inglês.
4. O [opus-mt-tc-big-en-pt](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-pt) traduz para português do Brasil.

Frontend e backend são servidos pelo mesmo servidor, então não é preciso liberar CORS.

```
backend/
  app.py        rotas e validação do upload
  captioner.py  legenda e tradução
  wsgi.py       ponto de entrada do gunicorn
frontend/       HTML, CSS e JavaScript
tests/          testes com um captioner falso
```

## API

`POST /analyze-image` com `multipart/form-data` e o campo `image`.

```json
{ "description": "um gato sentado em um sofá" }
```

| Status | Quando |
|---|---|
| 400 | Nenhum arquivo enviado, arquivo que não é imagem ou resolução muito alta |
| 413 | Arquivo maior que o limite |

## Como rodar

Na primeira execução os modelos são baixados do Hugging Face, cerca de 1,9 GB no total.

```bash
pip install -r requirements.txt
cd backend
python app.py
```

Acesse http://localhost:5000.

Com Docker:

```bash
docker build -t descricao-de-imagens .
docker run -p 8000:8000 -v hf-cache:/home/app/.cache/huggingface descricao-de-imagens
```

Acesse http://localhost:8000. O volume `hf-cache` guarda os modelos para não baixar de novo a cada execução.

| Variável | Padrão |
|---|---|
| `MAX_UPLOAD_MB` | `5` |
| `CAPTION_MODEL` | `Salesforce/blip-image-captioning-base` |
| `TRANSLATION_MODEL` | `Helsinki-NLP/opus-mt-tc-big-en-pt` |

## Testes

Os testes trocam o modelo por uma versão falsa e rodam sem PyTorch.

```bash
pip install -r requirements-dev.txt
pytest
```

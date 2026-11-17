# Classificador de Cachorros e Gatos

Classificador de imagens que diz se uma foto é de um cachorro ou de um gato. O modelo é uma ResNet-18 ajustada com transfer learning e roda numa interface feita com Streamlit.

Demo: https://classificador-imagens.streamlit.app

## Treinamento

O treino foi feito no Google Colab com PyTorch, usando o dataset Dogs vs. Cats do Kaggle.

- As imagens foram separadas em pastas de treino e teste, e as imagens corrompidas foram removidas.
- O pré-processamento redimensiona para 224x224 e converte para tensor.
- A ResNet-18 pré-treinada no ImageNet teve a última camada trocada por uma saída de 2 classes.
- Otimizador Adam, função de perda Cross-Entropy, 5 épocas.

| Época | Loss |
|---|---|
| 1 | 0.1637 |
| 2 | 0.1116 |
| 3 | 0.0874 |
| 4 | 0.0775 |
| 5 | 0.0627 |

**Acurácia no conjunto de teste: 97,56%.** O treino levou cerca de 10 minutos no Colab.

O notebook original se perdeu. O `train.py` foi reescrito a partir dele com a mesma arquitetura, otimizador, função de perda e pré-processamento, e gera um arquivo compatível com o app. O tamanho do batch e a taxa de aprendizado do treino original não foram registrados, por isso o script usa valores padrão (32 e 0.001).

### Treinar de novo

Organize o dataset assim:

```
data/
  train/
    cats/
    dogs/
  test/
    cats/
    dogs/
```

```bash
pip install -r requirements.txt
python train.py --data data --epochs 5
```

## Como funciona a previsão

O app aplica o mesmo pré-processamento do treino e calcula a probabilidade de cada classe com softmax. Se a confiança ficar abaixo de 70%, ele responde que não tem certeza, em vez de forçar uma das classes.

## Limitações

O modelo só conhece duas classes. Uma foto de pessoa ou de outro animal vai cair em "cachorro" ou "gato", às vezes com confiança alta. O limite de 70% reduz esse problema, mas não resolve. Imagens muito escuras ou cortadas também podem ser classificadas errado.

## Como rodar

O modelo treinado (43 MB) fica nas [Releases](https://github.com/kaiquedu/visao-computacional-estudos/releases) do repositório e é baixado automaticamente na primeira execução.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Com Docker:

```bash
docker build -t classificador-cao-gato .
docker run -p 8501:8501 classificador-cao-gato
```

Acesse http://localhost:8501.

| Variável | Padrão | Descrição |
|---|---|---|
| `MODEL_PATH` | `modelo_dogs_vs_cats.pth` | Onde o modelo fica salvo |
| `MODEL_URL` | link da release `modelo-v1` | De onde o modelo é baixado |

## Testes

```bash
pip install -r requirements-dev.txt
pytest
```

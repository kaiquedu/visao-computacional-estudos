# Visão Computacional: Estudos

Projetos de estudo com modelos de visão computacional em Python. Cada pasta é independente e tem o próprio README, dependências, testes e Dockerfile.

| Projeto | O que faz | Stack |
|---|---|---|
| [classificador-cao-gato](classificador-cao-gato) | Classifica fotos como cachorro ou gato com um modelo treinado por transfer learning | PyTorch, ResNet-18, Streamlit |
| [descricao-de-imagens](descricao-de-imagens) | Gera uma descrição em português para qualquer imagem | Flask, BLIP, MarianMT |

Demo do classificador: https://classificador-imagens.streamlit.app

## Testes

Os testes usam modelos falsos, então rodam sem baixar pesos e sem GPU.

```bash
cd classificador-cao-gato
pip install -r requirements-dev.txt
pytest
```

```bash
cd descricao-de-imagens
pip install -r requirements-dev.txt
pytest
```

## Licença

MIT

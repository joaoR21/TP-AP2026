# Trabalho Prático de Aprendizagem Profunda (módulo 1)

## Constituição do Grupo

**Grupo 2 (MEI)**

* Gustavo Manuel Marinho Barros - pg61527
* Enzo Gabriel Barros Vieira - pg61518
* João Ricardo Oliveira Macedo - pg60274
* Jorge Duarte Araújo Pereira - pg60276

## Organização do Repositório

O repositório encontra-se estruturado da seguinte forma:

* **`Subm1/`**:  ontém os ficheiros relativos à 1ª Submissão.
  * `subm1-g2-MEI-A.ipynb` e `subm1-g2-MEI-A.csv`: notebook e resultados da implementação própria.
  * `subm1-g2-MEI-B.ipynb` e `subm1-g2-MEI-B.csv`: notebook e resultados da implementação em PyTorch.
* **`src/`**: contém os scripts com o código desenvolvido (ex.: extração de features, definição das arquiteturas dos modelos `ffnn.py`, `logistic_regression.py`, etc.).
* **`models/`**: pasta onde estão guardados os pesos dos modelos previamente treinados (`model_logreg.npz`, `model_dnn.pt`).
* **`vectorizers/`**: contém os transformadores de texto guardados (ficheiros `.pkl` com os vetorizadores).
* **`datasets/`**: pasta utilizada para armazenar os dados de treino, validação e teste.
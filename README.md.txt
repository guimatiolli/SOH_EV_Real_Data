# SOH Estimation from Real Electric Vehicle Charging Events

**Author:** Guilherme Matiolli

**Institution:** Federal University of ABC (UFABC)

**Research Area:** Battery Management Systems (BMS), Machine Learning and State of Health (SOH) Estimation

**Status:** Research Project


# SOH Estimation from Real Electric Vehicle Charging Events

Pipeline para preparação do dataset, balanceamento com SMOTER, treinamento e avaliação de modelos de Machine Learning para estimativa do **State of Health (SOH)** de baterias de veículos elétricos utilizando eventos reais de carregamento.

---

# Objetivo

Este projeto implementa um pipeline completo para treinamento e avaliação de modelos de regressão e Mixture of Experts (MoE) para estimativa do SOH de baterias de veículos elétricos a partir de eventos de carregamento.

O fluxo contempla:

- preparação do Dataset Gold;
- validação cruzada estratificada em 5 folds;
- balanceamento do conjunto de treinamento utilizando SMOTER;
- treinamento de modelos globais e especialistas;
- treinamento de um Gate baseado em XGBoost;
- inferência em dados não vistos;
- consolidação das predições Out-of-Fold (OOF);
- avaliação por RMSE global e por faixas de SOH.

---

# Estrutura do projeto

```
SOH_EV_Real_Data/
│
├── CONFIG/
│   ├── gate_80_features_audit.xlsx
│   └── ...
│
├── DATA/
│   └── Dataset Gold
│
├── NOTEBOOK/
│   └── SOH_BALANCE_DATASET_V2.ipynb
│
├── OUTPUT/
│   ├── audit/
│   ├── folds/
│   ├── metrics/
│   ├── models/
│   └── predictions/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Dataset

O projeto utiliza o **Dataset Gold**, construído a partir do conjunto disponibilizado no NeurIPS 2023 Battery Dataset.

Cada amostra representa um **evento completo de carregamento**, contendo:

- metadados do veículo;
- quilometragem;
- aproximadamente 250 features físicas extraídas do perfil de carregamento;
- SOH de referência calculado utilizando a metodologia Top-10.

---

# Pipeline

O fluxo executado pelo notebook é:

```
Dataset Gold

↓

Limpeza

↓

Remoção de SOH < 0.92

↓

5-fold Stratified Cross Validation

↓

SMOTER
(apenas no conjunto de treino)

↓

Oracle Labels (OOF)

↓

Treinamento do Gate

↓

Treinamento dos Especialistas

↓

Inferência

↓

Predições OOF

↓

RMSE Global

↓

RMSE por faixa de SOH
```

---

# Modelos utilizados

## Modelo Global

- ExtraTreesRegressor

Features:

- 20 features físicas
- mileage

Total:

- **21 features**

---

## Especialistas

### Especialista A

- XGBoost Regressor
- SOH entre 0.920 e 0.945

---

### Especialista B

- ExtraTrees Regressor
- SOH entre 0.940 e 0.975

---

### Especialista C

- ExtraTrees Regressor
- SOH entre 0.970 e 1.005

Todos utilizam:

- 20 features físicas
- mileage

---

## Gate

Modelo:

- XGBoost Classifier

Entradas:

- 75 features selecionadas
- mileage

Total:

- **76 features**

---

# Estratégia de Validação

O notebook utiliza validação cruzada estratificada em cinco folds.

Para evitar vazamento de dados:

- o SMOTER é aplicado somente no treino de cada fold;
- as labels do Gate são obtidas por Out-of-Fold (OOF);
- a avaliação final é realizada apenas no conjunto de validação externo de cada fold.

---

# Métrica principal

O desempenho dos modelos é avaliado utilizando:

- RMSE Global
- RMSE por faixas de SOH

As predições de todos os folds são concatenadas para formar uma única avaliação OOF.

---

# Como executar

Clone o repositório:

```bash
git clone <repository>
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Abra o notebook:

```
NOTEBOOK/SOH_BALANCE_DATASET_V2.ipynb
```

Execute todas as células em sequência.

---

# Saídas geradas

Durante a execução são produzidos:

- auditorias do dataset;
- folds balanceados;
- modelos treinados;
- métricas;
- predições;
- tabelas de comparação entre especialistas e modelo global.

Todos os arquivos são gravados automaticamente na pasta:

```
OUTPUT/
```

---

# Dependências principais

- Python 3.11+
- NumPy
- Pandas
- Scikit-Learn
- XGBoost
- SMOGN
- Matplotlib
- Joblib
- OpenPyXL

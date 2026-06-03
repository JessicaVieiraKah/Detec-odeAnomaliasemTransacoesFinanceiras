# 💳 Detecção de Anomalias em Transações Financeiras com Machine Learning

## 📌 Sobre o Projeto

Este projeto tem como objetivo detectar transações fraudulentas em cartões de crédito utilizando técnicas de Machine Learning em Python.

Foram aplicadas diferentes abordagens para lidar com o problema de classificação desbalanceada, incluindo:

* Regressão Logística
* Random Forest
* XGBoost
* Undersampling
* Oversampling com SMOTE
* Ajuste de Threshold
* Otimização de Hiperparâmetros com GridSearchCV
* Interpretabilidade de modelos com SHAP

O conjunto de dados utilizado contém transações reais anonimizadas, amplamente utilizado em estudos de detecção de fraudes.

---

## 🎯 Objetivos

* Identificar transações fraudulentas com alta precisão.
* Comparar diferentes algoritmos de classificação.
* Tratar o desbalanceamento das classes.
* Avaliar modelos através de métricas apropriadas.
* Interpretar as decisões do modelo utilizando Explainable AI (XAI).

---

## 📊 Dataset

O dataset foi obtido diretamente do TensorFlow:

https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

### Informações

* Total de registros: 284.807
* Variáveis: 31
* Classe alvo:

  * 0 → Transação Normal
  * 1 → Fraude

O dataset apresenta forte desbalanceamento, com menos de 1% das transações classificadas como fraude.

---

## 🛠 Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* SHAP
* Matplotlib
* Imbalanced-Learn (SMOTE)

---

## 🔄 Etapas do Projeto

### 1. Carregamento dos Dados

Leitura do dataset utilizando Pandas.

### 2. Análise Inicial

Verificação da distribuição das classes para identificar o desbalanceamento dos dados.

### 3. Engenharia de Features

Transformação logarítmica da variável Amount:

```python
df["Amount_log"] = np.log1p(df["Amount"])
```

Padronização dos valores monetários:

```python
StandardScaler()
```

### 4. Separação dos Dados

Divisão em conjuntos de treino e teste utilizando:

```python
train_test_split()
```

com estratificação das classes.

---

## 🤖 Modelos Treinados

### Regressão Logística

Modelo baseline para classificação binária.

### Random Forest

Configurações utilizadas:

* 50 árvores
* Profundidade máxima: 10
* Balanceamento automático das classes

### XGBoost

Configurações utilizadas:

```python
XGBClassifier(
    scale_pos_weight=10,
    eval_metric="logloss"
)
```

---

## ⚖ Tratamento do Desbalanceamento

### Undersampling

Redução da quantidade de exemplos da classe majoritária.

### Oversampling (SMOTE)

Geração sintética de novas amostras da classe minoritária.

```python
SMOTE()
```

---

## 📈 Avaliação dos Modelos

Foram utilizadas as seguintes métricas:

* Precision
* Recall
* F1-Score
* ROC-AUC
* Curva ROC
* Curva Precision-Recall

### Relatório de Classificação

```python
classification_report()
```

### Curva ROC

Avalia a capacidade de separação entre classes.

### Curva Precision-Recall

Especialmente importante para bases desbalanceadas.

---

## 🎛 Ajuste de Threshold

Foi realizado ajuste manual do limiar de decisão:

```python
threshold = 0.3
```

Objetivo:

* Aumentar o Recall
* Reduzir falsos negativos

Em problemas de fraude, deixar uma fraude passar costuma ser mais custoso do que gerar um falso alerta.

---

## 🔍 Otimização de Hiperparâmetros

Utilização do GridSearchCV para encontrar a melhor configuração do XGBoost.

Parâmetros avaliados:

```python
{
    "max_depth": [3, 5],
    "n_estimators": [50, 100]
}
```

Métrica de otimização:

```python
scoring="recall"
```

---

## 📊 Importância das Variáveis

Análise da relevância das features através do XGBoost:

```python
xgb.feature_importances_
```

Visualização utilizando Matplotlib.

---

## 🧠 Explainable AI (XAI)

O projeto utiliza SHAP para explicar as decisões do modelo.

```python
import shap

explainer = shap.Explainer(xgb)

shap_values = explainer(x_test[:100])

shap.plots.bar(shap_values)
```

Benefícios:

* Interpretabilidade
* Transparência
* Identificação das variáveis mais relevantes
* Maior confiança nas previsões do modelo

---
---

## 🚀 Como Executar

### Clonar o Repositório

```bash
git clone https://github.com/JessicaVieiraKah/DeteccaodeAnomaliasemTransacoesFinanceiras.git
```

### Entrar na Pasta

```bash
cd fraud-detection
```

### Instalar Dependências

```bash
pip install -r requirements.txt
```

### Executar o Notebook

```bash
jupyter notebook
```

---

## 📚 Aprendizados

Durante o desenvolvimento deste projeto foram praticados conceitos como:

* Machine Learning supervisionado
* Classificação binária
* Tratamento de dados desbalanceados
* Engenharia de atributos
* Avaliação de modelos
* Explainable AI (XAI)
* Otimização de hiperparâmetros

---

## 👩‍💻 Autora

Jéssica Vieira da Rosa

Estudante de Engenharia de Software com foco em:

* Ciência de Dados
* Machine Learning
* Inteligência Artificial
* Desenvolvimento de Software


⭐ Se este projeto foi útil para você, considere deixar uma estrela no repositório.

# 🛍️ Retail Churn Intelligence & Predictive Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange.svg)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey.svg)](https://www.sqlite.org/)

Sistema analítico completo desenvolvido para o setor de varejo, unindo **Engenharia de Dados (SQLite/SQL)**, **Modelagem RFM (Recência, Frequência e Valor Monetário)**, **Machine Learning (Random Forest)** e **Deploy de Aplicação Web Interativa (Streamlit)**.

---

## 🚀 Sobre o Projeto

No varejo moderno, reter clientes é significativamente mais lucrativo do que adquirir novos. Este projeto resolve o problema de negócio de **Previsão de Churn (Evasão de Clientes)**, permitindo que gestores comerciais identifiquem compradores em risco de abandono antes que a perda da receita ocorra.

---

## 🛠 Arquitetura e Tecnologias Utilizadas

* **Linguagem:** Python
* **Banco de Dados & ETL:** SQLite, Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest Classifier, Train-Test Split, Métricas de Avaliação)
* **Interface & Visualização:** Streamlit (Dashboard interativo com métricas e simulador de risco em tempo real)

---

## 📊 Principais Funcionalidades

1. **Pipeline de Dados Automatizado:** Criação e população de um banco relacional SQLite simulando transações reais de varejo.
2. **Cálculo de Métricas RFM:** Agrupamento e engenharia de atributos comportamentais de clientes por meio de consultas e agregações em Pandas/SQL.
3. **Modelo Preditivo Supervisionado:** Classificação binária utilizando *Random Forest* para pontuar a probabilidade de churn com alta acurácia.
4. **Simulador Executivo Web:** Interface em Streamlit onde o usuário simula perfis de clientes e recebe instantaneamente o diagnóstico de risco e sugestões de ações comerciais.

---

## ⚙️ Como Executar o Projeto Localmente

Para rodar este projeto em sua máquina, siga os passos abaixo:

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/ThayroneConstancio/retail-churn-intelligence.git](https://github.com/ThayroneConstancio/retail-churn-intelligence.git)

   Entre na pasta do projeto:

1. Entre na pasta do projeto:
cd retail-churn-intelligence
Instale as dependências necessárias:

2. Instale as dependências necessárias:
pip install streamlit pandas numpy scikit-learn

3. Execute a aplicação Streamlit:
streamlit run app.py

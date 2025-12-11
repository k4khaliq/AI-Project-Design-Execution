# 📘 Employee Wellbeing & Burnout Early Warning System

### ML-Powered Burnout Prediction + RAG-Enhanced AI Advisor

*A Streamlit application for predicting burnout risk, explaining drivers, and providing HR-policy-aware AI guidance.*

---

# 📂 Table of Contents

* [🌟 Project Overview](#-project-overview)
* [🧰 Requirements](#-requirements)
* [💻 Installation Guide](#-installation-guide)

  * [1. Install Python (Windows & macOS)](#1-install-python-windows--macos)
  * [2. Clone This Repository](#3-clone-this-repository)
  * [3. Create and Activate Virtual Environment](#4-create-and-activate-virtual-environment)
  * [4. Install Python Dependencies](#5-install-python-dependencies)
* [🤖 Installing & Running Ollama (Local LLM Engine)](#-installing--running-ollama-local-llm-engine)

  * [Install Ollama](#install-ollama)
  * [Download Required Model](#download-required-model)
  * [Test the Model](#test-the-model)
* [🚀 Run the Application](#-run-the-application)
* [📦 Project Structure](#-project-structure)
* [🛠 Troubleshooting](#-troubleshooting)
* [📜 License](#-license)

---

# 🌟 Project Overview

This project predicts employee burnout using machine learning and provides:

* **Burnout risk scoring (0–100%)**
* **Risk classification: Low / Medium / High**
* **Explainability**: Top drivers (workload, stress, support, recognition, sleep, job satisfaction)
* **Individual analysis dashboard**
* **RAG-powered AI Advisor** with:

  * Action Playbooks
  * Context-aware coaching chat
  * Notes for managers
  * HR-policy-driven recommendations
* **Department-level heatmaps & trend views**

The system runs 100% locally using **Ollama** for AI — no cloud LLM needed.

---

# 🧰 Requirements

✔ Windows 10/11 or macOS
✔ Python **3.10 or newer**
✔ Git
✔ Ollama (for running local LLMs)
✔ Minimum specs:

* 8 GB RAM (16 GB recommended)
* 10 GB free disk space

---

# 💻 Installation Guide

## 1. Install Python (Windows & macOS)

### **Windows**

Download Python from:
🔗 [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)

During installation, check:

☑ **Add Python to PATH**
☑ **Install pip**

Verify installation:

```bash
python --version
pip --version
```

### **macOS**

Install via Homebrew:

```bash
brew install python
```

Verify:

```bash
python3 --version
pip3 --version
```

---

## 4. Create and Activate Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 5. Install Python Dependencies

```
pip install -r requirements.txt
```

---

# 🤖 Installing & Running Ollama (Local LLM Engine)

## Install Ollama

### Windows:

Download installer:
[https://ollama.com/download/windows](https://ollama.com/download/windows)

### macOS:

```bash
brew install ollama
```

or download DMG:
[https://ollama.com/download/mac](https://ollama.com/download/mac)

Verify installation:

```bash
ollama --version
```

---

## Download Required Model

This app works best with **LLaMA 3.1 8B** or **Mistral 7B**.

Example (recommended):

```bash
ollama pull llama3.1:8b
```

Or:

```bash
ollama pull mistral:7b
```

---

## Test the Model

```bash
ollama run llama3.1:8b "Hello"
```

If you see a response → you're good!

---

# 🚀 Run the Application

Inside your virtual environment, run:

```bash
streamlit run app.py
```

The app opens at:

👉 [http://localhost:8501](http://localhost:8501)

You're ready to explore:

* Dashboard
* Predictions
* Individual Analysis
* AI Advisor
* HR Policy Assistant

---

# 📦 Project Structure

```
├── app.py                    # Main Streamlit Application
├── data_generation.py        # Synthetic/real data generation module
├── features.py               # Feature engineering & scoring
├── model_training.py         # ML model training pipeline
├── rag_engine.py             # HR Policy RAG Engine
├── llm_integration.py        # Ollama / OpenAI LLM wrapper
├── config.py                 # Configuration and model settings
├── hr_policies/              # HR policy text files for RAG
├── data/                     # Training and prediction datasets
├── requirements.txt          # Python dependencies
└── README.md                 # This file!
```

---

# 🛠 Troubleshooting

### ❌ **Ollama model not found**

```bash
ollama pull llama3.1:8b
```

### ❌ **Streamlit cannot find rerun()**

You installed an older version. Update:

```bash
pip install --upgrade streamlit
```

### ❌ **Python deps failing on Windows**

Update pip:

```bash
python -m pip install --upgrade pip
```

### ❌ **App shows blank page**

Check the terminal running Streamlit — look for missing imports or syntax errors.

### ❌ **No CUDA / GPU**

Ollama still works but slower. CPU mode is automatic.

---

# 📜 License

MIT License — free to modify, deploy, and customize.

---

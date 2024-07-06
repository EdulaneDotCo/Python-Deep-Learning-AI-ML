![Introduction to Python](https://res.cloudinary.com/dj2j9slz5/image/upload/v1720270606/Python-AI-ML-Session-01_rphgcl.jpg)

# Introduction to Python

Python is a high-level, interpreted programming language known for its simplicity and readability. Developed by Guido van Rossum and first released in 1991, Python emphasizes code readability and allows programmers to express concepts in fewer lines of code compared to languages like C++ or Java.

---

## Table of Contents
- [Introduction to Python](#introduction-to-python)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
    - [1. Download Python](#1-download-python)
    - [2. Install Python](#2-install-python)
  - [IDEs](#ides)
  - [Jupyter notebook installation](#jupyter-notebook-installation)
    - [Installation with Anaconda](#installation-with-anaconda)
      - [1. Download Anaconda](#1-download-anaconda)
      - [2. Install Anaconda](#2-install-anaconda)
      - [3. Verify Installation](#3-verify-installation)
      - [4. Launch Jupyter Notebook](#4-launch-jupyter-notebook)
    - [Alternative Installation using pip](#alternative-installation-using-pip)
      - [1. Install Jupyter Notebook](#1-install-jupyter-notebook)
      - [Recommended note](#recommended-note)
  - [Basic Syntax](#basic-syntax)
    - [1. Print Statement](#1-print-statement)
    - [2. Comments](#2-comments)

## Introduction
Python is a high-level, interpreted programming language known for its simplicity and readability. It is widely used in various fields such as web development, data science, artificial intelligence, and more.

## Installation
### 1. Download Python
- Visit the [official Python website](https://www.python.org/).
- Download the latest version of Python. Make sure to download the version appropriate for your operating system (Windows, macOS, or Linux).

### 2. Install Python
- **Windows:**
  - Run the downloaded executable file.
  - Make sure to check the "Add Python to PATH" option.
  - Follow the installation prompts.

- **macOS:**
  - Open the downloaded .pkg file and follow the installation prompts.
  - Alternatively, you can use Homebrew: `brew install python3`.

- **Linux:**
  - Use the package manager of your distribution. For example, on Ubuntu: 
    ```bash
    sudo apt update
    sudo apt install python3
    ```

## IDEs
Integrated Development Environments (IDEs) provide a comprehensive environment to write, run, and debug Python code. Here are some popular IDEs:

1. **PyCharm:** A powerful IDE specifically for Python, offering code analysis, graphical debugging, and more.
2. **Visual Studio Code:** A lightweight, open-source code editor with extensions for Python support.
3. **Jupyter Notebook:** An interactive web application for running and sharing Python code in a notebook format.
4. **Spyder:** An IDE tailored for data science, integrating with libraries such as NumPy, SciPy, and Matplotlib.

---

## Jupyter notebook installation

### Installation with Anaconda

#### 1. Download Anaconda

- Visit the [Anaconda download page](https://www.anaconda.com/products/individual).
- Choose the appropriate installer for your operating system (Windows, macOS, Linux).
- Download the installer.

#### 2. Install Anaconda

- **Windows:**
  - Double-click the downloaded `.exe` file.
  - Follow the prompts in the Anaconda Installer window.
  - Select "Install for me only".
  - Check "Add Anaconda to my PATH environment variable".

- **macOS:**
  - Double-click the downloaded `.pkg` file.
  - Follow the prompts in the Anaconda Installer window.

- **Linux:**
  - Open a terminal.
  - Navigate to the directory where the Anaconda installer was downloaded.
  - Run the following command:
    ```bash
    bash Anaconda3-2022.02-Linux-x86_64.sh
    ```
  - Follow the prompts in the installer.

#### 3. Verify Installation

- Open a terminal (or Anaconda Prompt on Windows).
- Type:
  
  ```bash
  conda --version
  ```
- This should display the version of conda installed, confirming that Anaconda is installed correctly.

#### 4. Launch Jupyter Notebook
  - Open a terminal (or Anaconda Prompt on Windows).
  - Type:
    ```bash
    jupyter notebook
    ```
  - This will open Jupyter Notebook in your default web browser.
  - You can create new notebooks, open existing ones, and start coding using Python.
  
### Alternative Installation using pip
If you already have Python installed and prefer to use `pip`:
#### 1. Install Jupyter Notebook
- Open a terminal (or command prompt on Windows).
- Type:
    ```bash
    pip install notebook
    ```
- This will also open Jupyter Notebook in your default web browser.

#### Recommended note
Installing Jupyter Notebook via Anaconda is recommended for beginners as it simplifies the setup process and ensures compatibility with other data science libraries included in Anaconda. Alternatively, installing with pip is suitable if you already have Python installed and prefer a minimal setup.

---

## Basic Syntax
Python syntax is designed to be readable and straightforward. Here are some fundamental elements:

### 1. Print Statement
```python
print("Hello, World!")
```

### 2. Comments
 - Single-line comment:
    ```python
      # This is a comment
    ```
 - Multi-line comment:
    ```python
      """
      This is a
      multi-line comment
      """
    ```
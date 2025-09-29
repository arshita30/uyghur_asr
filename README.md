# Uyghur Automatic Speech Recognition (ASR) with Whisper

This repository contains the code for a personal project focused on building an Automatic Speech Recognition (ASR) system for the Uyghur language. The goal was to explore the effectiveness of fine-tuning state-of-the-art models on low-resource languages and to create a functional speech-to-text pipeline from scratch.

The model is built by fine-tuning `openai/whisper-small` using the Hugging Face `transformers` and `datasets` libraries.

---

## 🚀 Project Overview

* **Objective:** To build a high-performance ASR system for Uyghur by fine-tuning the OpenAI Whisper model.
* **Model:** Fine-tuned `openai/whisper-small`.
* **Dataset:** This project uses a publicly available 23-hour Uyghur speech dataset (originally compiled for the "NPPE-2: Uyghur ASR" Kaggle competition).
* **Evaluation Metric:** The model's performance is measured by the **Character Error Rate (CER)**, where a lower score indicates higher accuracy.

---

## ✨ Key Features

* **End-to-End Pipeline:** A complete script for data loading, preprocessing, training, evaluation, and inference.
* **Hugging Face Integration:** Leverages the `Trainer` API for efficient training and evaluation workflows.
* **Audio Preprocessing:** Converts raw audio into log-Mel spectrograms as required by the Whisper model architecture.
* **Custom Data Collator:** Implements a data collator to dynamically pad sequences for efficient batching of audio and text data.
* **Inference Ready:** Generates transcriptions for new audio files after the model has been trained.

---

## 🛠️ Technologies & Libraries Used

* **Python 3.x**
* **PyTorch**
* **Hugging Face Transformers**: For the Whisper model and training infrastructure.
* **Hugging Face Datasets**: For efficient data handling and preprocessing.
* **Hugging Face Evaluate**: For calculating the CER metric.
* **Librosa**: For audio processing.
* **Pandas**: For data manipulation.

---

## 📦 Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arshita30/uyghur_asr
    cd uyghur_asr
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Setup:**
    * The complete dataset is included within this repository.
    * The script is already configured with the correct relative paths to use this data.
    * After cloning the repository, no data downloads or path configuration changes are required. The project is ready to run.
---

## ⚙️ How to Run

This project is a Jupyter Notebook (`.ipynb`) file. You can modify key parameters like `MODEL_CHECKPOINT`, `BATCH_SIZE`, and `NUM_TRAIN_EPOCHS` in the configuration cells near the top of the notebook.

There are two main ways to run it:

#### 1\. Interactively using Jupyter (Recommended)

This is the standard approach for notebooks. It allows you to run code cell-by-cell and inspect the outputs.

**Open your terminal in the project directory and run:**

```bash
jupyter lab uyghur_asr.ipynb
```

*(You can also use `jupyter notebook uyghur_asr.ipynb` for the classic interface).*

Once the notebook opens in your browser, you can execute all cells by selecting **"Cell" \> "Run All"** from the menu.

#### 2\. From the Command Line

This method executes the entire notebook from start to finish without opening the interactive interface, which is useful for automation.

```bash
jupyter nbconvert --to notebook --execute uyghur_asr.ipynb
```

-----

The notebook will perform the following steps:

1.  Load and preprocess the training and test data.
2.  Initialize the Whisper model and its processor.
3.  Set up the `Seq2SeqTrainer` for training.
4.  Train the model and save checkpoints.
5.  Perform inference on the test data.
6.  Generate a `submission.csv` file with the final transcriptions.
---

## 📈 Results

The fine-tuned model achieved a Character Error Rate (CER) of **0.12** on the held-out test set. This result demonstrates the viability of transfer learning for achieving strong performance in low-resource ASR tasks, turning a large, general-purpose model into a specialized tool for a specific language.

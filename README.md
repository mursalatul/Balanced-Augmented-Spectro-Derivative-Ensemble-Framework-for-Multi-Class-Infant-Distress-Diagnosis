
# Infant Cry Detection Model

## Table of Contents

1. [About the Model](#about-the-model)
2. [File Information](#file-information)
3. [Installation Process](#installation-process)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Dependencies](#dependencies)
8. [Authors](#authors)
9. [License](#license)
10. [Contact](#contact)

---

## About the Model

The **Infant Cry Detection Model** is a machine learning system designed to analyze infant cries and predict their underlying causes, such as hunger, discomfort, or tiredness, with an impressive accuracy of **98.67%**. 

Built using advanced audio feature extraction techniques with **Librosa** and a robust ensemble of classifiers (including **Random Forest**, **XGBoost**, and **SVM**), this model processes `.wav` audio files to provide reliable predictions.

This system is aimed at assisting **parents**, **caregivers**, and **healthcare professionals** by offering insights into an infant's needs, fostering better care and understanding.

---

## File Information

| File Name            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `main.py`            | Main script that balances data, trains models, evaluates performance, and saves the best model. |
| `models.py`          | Contains model definitions, training functions, evaluation metrics, and model utilities. |
| `voice_processing.py`| Handles audio preprocessing, MFCC extraction, data balancing, and augmentation techniques. |
| `requirements.txt`   | Lists all required Python dependencies.                                     |
| `README.md`          | Project documentation and setup instructions.                              |

---

## Installation Process

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mursalatul/cry-prediction.git
   cd cry-prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install `ffmpeg` if not already installed:**
   - If you're using conda:
     ```bash
     conda install ffmpeg
     ```

6. **(Optional)** Install necessary Python extensions for **VS Code** if you're using it as your IDE.

7. **Download and place the dataset:**
   - Dataset link: [DonateACry Corpus](https://github.com/gveres/donateacry-corpus)
   - Use the folder `donateacry_corpus_cleaned_and_updated_data`
   - Copy all category folders into the root of your project directory.

---

## Usage

To run the complete training and evaluation pipeline:

```bash
python main.py
```

It will:
- Preprocess and augment the data.
- Train multiple models.
- Display performance metrics.
- Save the best model and scaler in a `.pkl` file.

---

## Dataset

- **Dataset Source:** [DonateACry Corpus](https://github.com/gveres/donateacry-corpus)
- **Used Folder:** `donateacry_corpus_cleaned_and_updated_data`
- **Categories:**

| Data Folder Name | Number of Audio Files |
|------------------|------------------------|
| belly_pain       | 16                     |
| burping          | 8                      |
| discomfort       | 27                     |
| hungry           | 382                    |
| tired            | 24                     |

Due to the imbalance in original dataset, we applied **data augmentation** to ensure all classes are fairly represented during training.

---

## Model Training and Evaluation

We evaluated **27 different machine learning algorithms**. The top-performing models are listed below:

- ✅ **Random Forest**
- ✅ **XGBoost**
- ✅ **Support Vector Machine (SVM)**

### Sample Output:

```
📦 Preparing balanced dataset with augmentation...

📊 Class distribution after augmentation:
belly_pain: 382
burping: 382
discomfort: 382
hungry: 382
tired: 382

🔹 Train size: 1528, Test size: 382
🔹 Feature size: 40

🚀 Training Random Forest...
Random Forest: Accuracy=98.67%  Precision=98.70%  Recall=98.65%  F1=98.67%

📈 Final Model Metrics (sorted by accuracy):
Random Forest: Acc=98.67%  Prec=98.70%  Rec=98.65%  F1=98.67%

💾 Best model (Random Forest) and scaler saved to 'random_forest_model.pkl'
```

---

## Dependencies

- Python 3.10+
- numpy
- librosa
- pydub
- scikit-learn
- xgboost
- soundfile
- tqdm
- ffmpeg (external dependency)

Refer to `requirements.txt` for the full list.

---

## Authors

The following individuals contributed to the research, development, and implementation of this project:

- 🎓 [**Md. Tahzib Ul Islam**](#)  
  *Chairman, Dept. of CSE, Dhaka International University*

- 👨‍💻 [**Md Mursalatul Islam Pallob**](https://www.linkedin.com/in/mursalatul/)  
  *Researcher & Developer*

- 👩‍💻 [**Nahid Usha**](#)  
  *Researcher & Developer*

- 👩‍🔬 [**Farnaz Bashir Orpita**](#)  
  *Researcher*

- 👨‍🔧 [**Nahid Hasan**](#)  
  *Researcher*

- 👩‍🏫 [**Maliha Nusrat Tahira**](#)  
  *Researcher*

We are grateful to our advisor and all contributors for their valuable guidance and support throughout this project.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For any queries or collaborations, contact:

📧 mursalatulislam@gmail.com  
🔗 [GitHub Repository](https://github.com/mursalatul/cry-prediction)

---

## 📄 Research Paper

For more details on the methodology, dataset analysis, and evaluation, please read the full research paper:  
👉 [View Paper on ResearchGate / Google Drive / Journal Link Here](https://your-paper-link.com) *(replace with actual link)*

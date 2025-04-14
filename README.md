# Infant Cry Detection Model


## Table of Contents

1. [About the Model](#about-the-model)
2. [File Information](#file-information)
3. [Installation Process](#installation-process)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Dependencies](#dependencies)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

---

### About the Model

The Infant Cry Detection Model is a machine learning system designed to analyze infant cries and predict their underlying causes, such as hunger, discomfort, or tiredness, with an impressive accuracy of 98.67%. Built using advanced audio feature extraction with Librosa and a robust ensemble of classifiers (including Random Forest, XGBoost, and SVM), this model processes audio inputs to provide reliable predictions. The project aims to assist parents, caregivers, and healthcare professionals by offering insights into an infant's needs, fostering better care and understanding.

### File Information

- `infant_cry_detection_HIGH_ACCURACY.py`: Script for training the machine learning models and view accuracy.
- `infant_cry_detection_HIGH_ACCURACY_notebookv.ipynb`: same as infant_cry_detection_HIGH_ACCURACY.py by it is a nootbook version.
- `requirements.txt`: required Python packages.
- `accuracy_proof.csv`: a csv file containing the result of 500 times model train. Highest accuracy holder model name and the acccuracy number is stored here.

### Installation Process
1. Clone the repository:  
   ```bash
   git clone https://github.com/mursalatul/cry-prediction.git
   ```

2. Create a virtual environment:  
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:  
   - **Windows**:  
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:  
     ```bash
     source venv/bin/activate
     ```

4. Install the requirements:  
   ```bash
   pip install -r requirements.txt
   ```
   make sure to install `ffmpeg`. suppose you are using conda, then the comment will be
   ```bash
   conda install ffmpeg
   ```

5. Ensure you have the necessary extensions installed in **VS Code** to run Jupyter notebooks.

6. Download the dataset from [this link](https://github.com/gveres/donateacry-corpus).  
   Copy the folders from the <b>donateacry_corpus_cleaned_and_updated_data</b> directory and place them where you cloned the repository.


### Usage
- Run : `python infant_cry_detection_HIGH_ACCURACY.py`
it may need some time to precess the data, train the model and show you the result.

### Dataset
- Dataset link: [donatecry-corpus](https://github.com/gveres/donateacry-corpus)
- Dataset format: The dataset has 3 folder. Among them we will use the <b>donateacry_corpus_cleaned_and_updated_data</b> as it has all the cleaned data. The contants are:
<table>
   <tr>
      <td>Data Folder Name</td>
      <td>Number of Data(.wav audio)</td>
   </tr>
   <tr>
      <td>belly_pan</td>
      <td>16</td>
   </tr>
   <tr>
      <td>burping</td>
      <td>8</td>
   </tr>
   <tr>
      <td>discomfort</td>
      <td>27</td>
   </tr>
   <tr>
      <td>hungry</td>
      <td>382</td>
   </tr>
   <tr>
      <td>tired</td>
      <td>24</td>
   </tr>
</table>

### Model Training and Evaluation
We have worked with 8 types of machine learning algorithms to predict the cause of crying. Below are the average, highest, and lowest accuracies:


### Dependencies
[List key dependencies or refer to requirements.txt, e.g.,]
- Python 3.10 or above
- See `requirements.txt` for full list of packages (`numpy`, `librosa`, `scikit-learn`, `xgboost`, `pydub`).

### Contributing
[Invite contributions, e.g., "Contributions are welcome! Please submit a pull request or open an issue to discuss improvements."]

### License
[Specify the license, e.g., "This project is licensed under the MIT License. See the `LICENSE` file for details."]

### Contact

    Banknote Authentication Using Self-Organizing Map (SOM)


Problem Statement:
------------------
The goal of this project is to classify banknotes as authentic or fraudulent using the Self-Organizing Map (SOM) algorithm.
The SOM will help visualize and cluster banknotes based on their features, identifying patterns that distinguish genuine from fake banknotes.

Dataset:
--------
- Name: Banknote Authentication Dataset
- Source: UCI Machine Learning Repository
- Features:
    1. Variance of Wavelet Transformed Image (numeric)
    2. Skewness of Wavelet Transformed Image (numeric)
    3. Curtosis of Wavelet Transformed Image (numeric)
    4. Entropy of Image (numeric)
- Target:
    - 0: Fake
    - 1: Genuine
- Size: 1,372 instances
The dataset can be found at the following link: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

Software Requirements:
-------------
-Python 3.x

-Required libraries: numpy, matplotlib, sklearn, and sompy. You can install these libraries using:
    
     pip install numpy matplotlib sklearn sompy

Hardware Requirements:
--------------
-Any machine with a minimum of 4 GB RAM.

-An IDE or text editor (e.g., PyCharm, VSCode) for running the code.

Instructions:
-------------
1. Ensure that 'data_banknote_authentication.txt' is in the same directory as this script or
   and change the filepath accordingly
3. Install the required libraries using:
   
            pip install numpy pandas matplotlib scikit-learn minisom
4. Run the script using:

             python som_banknote_authentication.py


How to setup and Run the Code:
-------------
1. Clone the Repository:

         git clone https://github.com/kothapalliAnusha/SOM_Banknote_Authentication.git
         
   Alternatively, you can download the repository as a ZIP file from GitHub and extract it to your desired location.
2. Navigate to Project Directory:

       cd SOM_Banknote_Authentication
4. Set Up Virtual Environment:

          python -m venv env
        env\Scripts\activate
5. Install Dependencies:

        pip install -r requirements.txt
6. Run the Application:

        python som_banknote_authentication.py
7. View Results:
The output of the model will be displayed in the terminal.A visualization window will pop up displaying the SOM with clusters representing fake and genuine banknotes.

🧠 Customer Churn Prediction Using Artificial Neural Networks

Predicting which customers are at risk of leaving a business is a powerful way to help companies improve retention and boost profits. This deep learning project uses a custom-built Artificial Neural Network (ANN) to predict bank customer churn with remarkable accuracy.



🚀 Project Overview

	•	Goal: Build a machine learning model that predicts whether a bank customer will leave (churn) or stay, based on their data.
	•	Techniques: Uses a Multilayer Perceptron (MLP) implemented in Keras/TensorFlow.
	•	Dataset: Real-world bank customer data featuring attributes like geography, age, account balance, tenure, products, and more.
	•	Result: Achieves strong predictive performance, supporting banks and businesses in proactive customer retention strategies.


🛠️ Features

	•	End-to-end churn prediction pipeline
	•	Data preprocessing: handling missing values, encoding categorical variables, and feature scaling
	•	Custom ANN model with multiple hidden layers
	•	Model evaluation using accuracy, classification report, and confusion matrix
	•	Training and validation accuracy visualized over epochs
	•	Clear and reproducible code for experimentation and learning


📂 Project Structure

     .
     ├── data/               # (Add your dataset here)
     ├── images/             # (Store result charts & figures)
     ├── churn_model.py      # Core python code/model
     ├── README.md
     └── requirements.txt    # List of dependencies


⚡ Quick Start


	1.	Clone the repository
    git clone https://github.com/[your-username]/churn-prediction-ann.git
    cd churn-prediction-ann
    2.	Install dependencies
    pip install -r requirements.txt
    3.	Add Dataset
	•	Place the `Churn_Modelling.csv` file in the `data/` folder.
	4.	Run the codepython churn_model.py
     python churn_model.py

🧑‍💻 How it Works


	•	Step 1: Load and preprocess the dataset (drop unnecessary columns, encode categorial variables, scale features)
	•	Step 2: Split data into training and test sets
	•	Step 3: Build a Sequential ANN using Keras, with ReLU activations and a sigmoid output neuron for binary classification
	•	Step 4: Train and validate the model, tracking accuracy
	•	Step 5: Evaluate the final model on the test set and visualize performance


📊 Sample Results


	•	Training Accuracy: ~86%
	•	Test Accuracy: ~84%
	•	Visualizations: Includes accuracy curves and confusion matrix

💡 Motivation

Customer retention is far more cost-effective than acquisition. By leveraging deep learning, banks and businesses can better understand the drivers of churn, anticipate risk, and act before it’s too late. This project demonstrates a proven approach that is easily extendable and adaptable to a wide range of tabular data problems.

📝 Contributing
Found a bug, or want to make this project even better? Pull requests and suggestions are very welcome!

📜 License
This project is licensed under the MIT License.
Let’s predict churn—and help businesses grow—using the power of deep learning!



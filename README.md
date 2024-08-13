![banner](https://raw.githubusercontent.com/teixeirazeus/Spam-Doggy/main/readme_assets/banner.png)
![License](https://raw.githubusercontent.com/teixeirazeus/Spam-Doggy/dbd4392b71abfbd0b4b717256d1fcb71c1b7dec6/readme_assets/mit.svg)

# Spam Doggy
Spam Doggy is a spam classifier that uses a Naive Bayes classifier to classify emails as spam or not spam. It is designed to be easy to use and efficient, leveraging the power of machine learning to accurately identify unwanted emails.

## Features
- Utilizes a Naive Bayes classifier for efficient spam detection.
- Easy to train with custom datasets.
- Supports saving and loading models for quick predictions without retraining.

## Installation
To use Spam Doggy, you need to have Python installed along with some additional dependencies. You can install the necessary packages using:

```bash
pip install -r requirements.txt
```

## Usage

### train model
```python
from spam_doggy import SpamDoggy

# Load default training data
spam_doggy = SpamDoggy()
spam_doggy.set_default_training_data()

# Train the model
spam_doggy.train(print_report=True)

# Save the trained model and vectorizer
spam_doggy.save_model('vectorizer.joblib', 'classifier.joblib')
```

### load model
```python
# Create a new instance of SpamDoggy
spam_doggy = SpamDoggy()

# Load the pre-trained model and vectorizer
spam_doggy.load_model('vectorizer.joblib', 'classifier.joblib')

# Predict the class of a new email
email_text = "Congratulations! You've won a free cruise. Call now to claim your prize."
prediction = spam_doggy.predict(email_text)
print(f"The email is classified as: {prediction}")
```

### custom dataset
If you want to train the model with your own dataset, ensure your data is in a pandas.DataFrame with two columns: Message (the email text) and Category (the label: "spam" or "not spam").


```python
import pandas as pd

# Example custom data
data = {'Message': ["Free money", "Hi there, how are you?", "Limited time offer!"],
        'Category': ["spam", "not spam", "spam"]}

custom_df = pd.DataFrame(data)

# Train with custom data
spam_doggy = SpamDoggy()
spam_doggy.set_train_data(custom_df)
spam_doggy.train(print_report=True)
```
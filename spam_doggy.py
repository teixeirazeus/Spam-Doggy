from typing import Union
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from datasets.data_loader import load_ashfakyeafi


class SpamDoggy:
    """Spam classifer that uses bayens network to classify spam emails."""

    def __init__(self, train_with_default_data: bool = False):
        self.df = None
        if train_with_default_data:
            self.set_default_training_data()
            self.train()

    def set_train_data(self, df: DataFrame):
        """Set the training data for the classifier.

        Args:
            df (PandasDataFrame): The training data.
        """
        self.df = df

    def set_default_training_data(self):
        """Set the training data for the classifier.

        Args:
            df (PandasDataFrame): The training data.
        """
        self.df = load_ashfakyeafi()

    def train(self, print_report: bool = False):
        """Train the classifier."""

        # Dividir os dados em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            self.df["Message"], self.df["Category"], test_size=0.3, random_state=42
        )

        # Extrair características do texto
        vectorizer = CountVectorizer()
        X_train_counts = vectorizer.fit_transform(X_train)
        X_test_counts = vectorizer.transform(X_test)

        # Treinar o classificador Naive Bayes
        classifier = MultinomialNB()
        classifier.fit(X_train_counts, y_train)

        # Avaliar o classificador
        y_pred = classifier.predict(X_test_counts)
        accuracy = accuracy_score(y_test, y_pred)
        self.accuracy = accuracy

        if print_report:
            print(f"Accuracy: {accuracy:.2f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))

        return accuracy

    def predict(self, text: str):
        """Predict the class of a given text."""

        # Extrair características do texto
        vectorizer = CountVectorizer()
        X_counts = vectorizer.fit_transform(self.df["Message"])

        # Treinar o classificador Naive Bayes
        classifier = MultinomialNB()
        classifier.fit(X_counts, self.df["Category"])

        # Prever a classe do texto
        X_text_counts = vectorizer.transform([text])
        return classifier.predict(X_text_counts)[0]

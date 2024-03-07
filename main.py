import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
import sys

# This function displays instructions incase of erroneous parameters.
def message():
    print("Please provide the arguments in the following format: <path_to_dataset>")
    print("*** Note ***")
    print("Data in the folder should be arranged in the following categories: ")
    print('business, entertainment, politics, sport, tech')


def import_data(path):
    print("Importing data...")
    folder_list = ['business', 'entertainment', 'politics', 'sport', 'tech'] #List of folders
    train_documents = []
    fc = 0

    labels = []

    for folder in folder_list:
        folder_path = path + "/" + folder # Append path and folder sequentially for extraction.

        for file in os.listdir(folder_path):
            try:
                file_path = folder_path + "/" + file
                with open(file_path, 'r', encoding='ISO-8859-1') as f:
                    f_txt = f.read().replace('\n', '').lower()
                    train_documents.append(f_txt)
                    labels.append(fc)
            except:
                continue

        fc = fc + 1

    return train_documents, labels


def get_train_test_split(path):
    corpus, labels = import_data(path)
    print("Computing train test split...")
    X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42) # Train-test split

    return X_train, X_test, y_train, y_test


nlp = spacy.load("en_core_web_sm")

# This function will compute the NER vectors.
def get_ner_vec(X_train, X_test):
    print("Computing NER vectors...")

    def extract_entities(text):
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        return " ".join(entities)

    X_train_ner = [extract_entities(text) for text in X_train]
    X_test_ner = [extract_entities(text) for text in X_test]

    return X_train_ner, X_test_ner

# Get POS Tagging vectors
def get_pos_vec(X_train, X_test):
    print("Computing POS Tagging...")

    def extract_pos_tags(text):
        doc = nlp(text)
        pos_tags = [token.pos_ for token in doc]
        return " ".join(pos_tags)

    X_train_pos = [extract_pos_tags(text) for text in X_train]
    X_test_pos = [extract_pos_tags(text) for text in X_test]

    return X_train_pos, X_test_pos

# This function combines the tf-idf, POS tagging and NER vectors into one. So they can be used.
def get_combined_features(X_train, X_test, X_train_ner, X_test_ner, X_train_pos, X_test_pos):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    X_train_combined = X_train_tfidf + tfidf_vectorizer.transform(X_train_ner) + tfidf_vectorizer.transform(X_train_pos)
    X_test_combined = X_test_tfidf + tfidf_vectorizer.transform(X_test_ner) + tfidf_vectorizer.transform(X_test_pos)

    return X_train_combined, X_test_combined

# This function will do some feature selection and then perform the fit and calculate the metrics.
def run_learner(X_train_combined, X_test_combined, y_train, y_test):
    print("All initialized.. running learner...")

    k = int(X_train_combined.shape[1] / 100)  # Reduce the dimensionality by a factor of 100
    chi2_feature_selection = SelectKBest(chi2, k=k)

    X_train_chi2_selected = chi2_feature_selection.fit_transform(X_train_combined, y_train)
    X_test_chi2_selected = chi2_feature_selection.transform(X_test_combined)

    classifier = MultinomialNB()
    classifier.fit(X_train_chi2_selected, y_train)

    predictions = classifier.predict(X_test_chi2_selected)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

# Sequentially builds the program and runs it.
def init_run(path):
    X_train, X_test, y_train, y_test = get_train_test_split(path)
    X_train_ner, X_test_ner = get_ner_vec(X_train, X_test)
    X_train_pos, X_test_pos = get_pos_vec(X_train, X_test)
    X_train_combined, X_test_combined = get_combined_features(X_train, X_test,
                                                              X_train_ner, X_test_ner,
                                                              X_train_pos, X_test_pos)
    run_learner(X_train_combined, X_test_combined, y_train, y_test)


def main():
    if len(sys.argv) != 2:
        message()
        return

    path = sys.argv[1]
    if os.listdir(path) is None:
        message()
        print('Invalid path encountered')
    init_run(path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

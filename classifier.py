import pandas as pd
import numpy as np
from scipy.stats import norm

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Dict of attributes we consider, where key is the attribute name, value is bool - whether the data is continuous
attributes = {'Pclass': False, 'Sex': False, 'Age': True}
# 0 - not survived, 1 - survived
classes = [0, 1]


def fill_missing(data):
    """
    Fills missing values in attributes marked as continuous.
    Used values is the mean of all other rows.
    """
    for attr, is_processable in attributes.items():
        if not is_processable:
            continue
        data[attr].fillna(round(data[attr].mean()), inplace=True)


def normalize(data):
    """
    Normalizes data by converting string values to int.
    """
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})


def train(data):
    """
    Creates the table of all classes, with probabilities for each processable attribute for each value.
    """
    model = {}
    for c in classes:
        model[c] = {}
        class_set = data[data['Survived'] == c]
        class_count, _ = class_set.shape
        # Prob of the current class - P(c)
        class_prob = class_count / data.shape[0]

        for attr, is_continuous in attributes.items():
            model[c][attr] = {}

            # Process continuous values separately by storing they total mean and std,
            # we'll use later in the density function
            if is_continuous:
                # Store mean and unbiased sample std
                model[c][attr]['mean'] = class_set[attr].mean()
                model[c][attr]['std'] = class_set[attr].std()

                continue

            for value in data[attr].unique():
                # Set of rows of the current class, attribute having the current value
                attr_class_count, _ = class_set[class_set[attr] == value].shape
                # P(d | c) - conditional probability
                attr_class_prob = attr_class_count / class_count
                # (prob of value being in class c - likelihood * prob of class c - prior) / num of value - evidence
                # but we skip evidence, because it doesn't depend on class, so it's the same for all classes
                model[c][attr][value] = attr_class_prob * class_prob

    return model


def classify(row, model):
    """
    Classifies the row based on the model we trained before.
    Modifies the given row by setting 'Survived' attribute to the predicted class (0 - not survived, 1 - survived).
    """
    classes_probs = np.zeros(len(classes))
    for c in classes:
        prob = 1
        for attr, is_continuous in attributes.items():
            # For continuous attributes we calculate the probability using the prob density function
            # with the given mean and std of the attribute
            if is_continuous:
                prob *= norm.pdf(row[attr], model[c][attr]['mean'], model[c][attr]['std'])
                continue

            value = row[attr]
            prob *= model[c][attr][value]

        classes_probs[c] = prob
    row['Survived'] = np.argmax(classes_probs)

    return row


# Prepare train and test data (dataframes are mutable).
fill_missing(train_data)
normalize(train_data)
fill_missing(test_data)
normalize(test_data)

model = train(train_data)

# Apply the classification function to each row.
predicted = test_data.apply(classify, axis=1, model=model)
# Write the results.
predicted.to_csv('data/predicted.csv', columns=['PassengerId', 'Survived'], index=False)

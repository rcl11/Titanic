import pandas as pd
import tensorflow as tf

TRAIN_PATH = "train_1_corrected.csv"
#TRAIN_PATH = "train_corrected.csv"
#TEST_PATH = "test.csv"
TEST_PATH = "train_2_corrected.csv"
TEST_DATA_PATH = "test_corrected.csv"

CSV_COLUMN_NAMES = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Title','Deck']
CSV_COLUMN_NAMES_DATA = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Title','Deck']
SPECIES = ['Survived','Died']
NAN_VALUES = {'PassengerId':0, 'Survived':0,'Pclass':2,'Name':'','Sex':'male','Age':30,'SibSp':0,'Parch':0,'Ticket':'','Fare':10,'Cabin':'','Embarked':'','Title':'','Deck':''}


def load_data(y_name='Survived'):
    """Returns the Titanic dataset as (train_x, train_y), (test_x, test_y)."""

    train = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
    train.fillna(value=NAN_VALUES, inplace=True)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(TEST_PATH, names=CSV_COLUMN_NAMES, header=0)
    test.fillna(value=NAN_VALUES, inplace=True)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def load_test_data(y_name='Survived'):
     test = pd.read_csv(TEST_DATA_PATH, names=CSV_COLUMN_NAMES_DATA, header=0)
     test.fillna(value=NAN_VALUES, inplace=True)
     return test

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using a the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
    

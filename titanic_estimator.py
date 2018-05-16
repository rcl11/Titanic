#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Titanic dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import titanic_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = titanic_data.load_data()

    # Feature columns describe how to use the input.
    #my_feature_columns = []
    #for key in train_x.keys():
    #    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    my_feature_columns = [
        tf.feature_column.numeric_column(key='Pclass'),
        tf.feature_column.numeric_column(key='Age'),
        tf.feature_column.numeric_column(key='SibSp'),
        tf.feature_column.numeric_column(key='Parch'),
        tf.feature_column.numeric_column(key='Family_Size'),
        tf.feature_column.numeric_column(key='Fare'),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
            key='Sex',
            vocabulary_list=["male", "female"])),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
            key='Title',
            vocabulary_list=['Miss.', 'Mme.', 'Rev.', 'Dona.', 'Jonkheer.', 'Sir.', 'Mlle.', 'Mrs.', 'Capt.', 'Col.', 'Ms.', 'Mr.', 'Lady.', 'Dr.', 'the', 'Master.', 'Major.', 'Don.'])),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
            key='Deck',
            vocabulary_list=['A', 'C', 'B', 'E', 'D', 'G', 'F', 'T'])),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
            key='Embarked',
            vocabulary_list=["C", "Q", "S"]))
    ]    

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3)

    # Train the Model.
    classifier.train(
        input_fn=lambda:titanic_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:titanic_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    

    # Generate predictions from the model
    #expected = ['Setosa', 'Versicolor', 'Virginica']
    predict = titanic_data.load_test_data()
    #print(predict)

    predictions = classifier.predict(
        input_fn=lambda:titanic_data.eval_input_fn(predict,
                                                labels=None,
                                                batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%) for class ID: "{}"')
    template2 = ('{},{}\n')

    #for pred_dict, expec in zip(predictions, expected):
    outfile = open("predictions.csv","w")
    outfile.write("PassengerID,Survived\n")
    
    for counter,pred_dict in enumerate(predictions):
        #print(counter)
        #print(pred_dict)
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        #print(predict.iloc[counter,predict.columns.get_loc('PassengerId')], class_id)
        #print(template2.format(predict.iloc[counter,predict.columns.get_loc('PassengerId')],class_id))
        outfile.write(template2.format(predict.iloc[counter,predict.columns.get_loc('PassengerId')],class_id))
        #print(predict["PassengerID"])
        #print(template.format(titanic_data.SPECIES[class_id],
        #                      100 * probability, class_id))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

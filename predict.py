"""
Module for prediction of output using our decision tree model.
"""

import pickle

import numpy as np
import pandas as pd


def get_model(file_location):
    with open(file_location, 'rb') as f:
        return pickle.load(f)


def predict(input_data, model, columns, indices):
    """
    Predicts and returns the label.
    :param input_data: Input dictionary of input variables.
    :param model: Python dictionary model of our decision tree.
    :param columns: list of name of column fields
    :param indices: dictionary having information about indices of categorical and numeric variable
    :return: Predicted label.
    """

    if type(model) != dict:
        return model

    indicator = list(set(input_data.keys()) & set(model.keys()))

    # Using default value if nan(not a number) i.e missing value is encountered.
    if str(input_data[indicator[0]]) == 'nan':
        input_data[indicator[0]] = model[indicator[0]]['default']

    if indicator[0] in [columns[i] for i in indices['categorical']]:

        # if the indicator is categorical just do dictionary lookup
        return predict(input_data, model[indicator[0]][input_data[indicator[0]]], columns, indices)
    else:

        # if the indicator is numeric, get the comparator and do dictionary lookup in the appropriate format
        # loop for the dict_keys until it doesn't take 'default' value
        dict_keys = iter(model[indicator[0]].keys())
        comparator = next(dict_keys)
        while comparator == 'default':
            comparator = next(dict_keys)
        comparator = float(comparator[2:])

        # if the type is str, the value should have come as the replacement of missing data so let's convert it
        # into numeric form first depending on if it is greater or smaller
        if type(input_data[indicator[0]]) == str:
            if '>' in input_data[indicator[0]]:
                input_data[indicator[0]] = float(input_data[indicator[0]][2:]) + 1
            else:
                input_data[indicator[0]] = float(input_data[indicator[0]][2:]) - 1

        if input_data[indicator[0]] >= comparator:
            return predict(input_data, model[indicator[0]]['>=' + str(comparator)], columns, indices)
        else:
            return predict(input_data, model[indicator[0]]['< ' + str(comparator)], columns, indices)


if __name__ == "__main__":
    model = get_model('models/titanic.pickle')
    columns = model['columns']
    indices = model['indices']
    model = model['model']

    test_df = pd.read_csv('data/preprocessed/train.csv')
    test_y = test_df.values[:, 0]
    test_x = test_df.iloc[:, 1:]

    # create tuples of dictionaries from test data(which we splitted in constructor)
    test_tuples = test_x.T.to_dict().items()
    test_dict_list = [j for (i, j) in test_tuples]

    # predict the y values and save in an array
    y_pred = np.array([predict(i, model, columns, indices) for i in test_dict_list])
    print('Accuracy on test data:', 100 * sum((y_pred == test_y)) / y_pred.shape[0])

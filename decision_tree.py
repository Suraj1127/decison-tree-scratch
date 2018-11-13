#!/usr/bin/env python3

"""
Author: Suraj Regmi
Date: 8th November, 2018
Description: Implementation of decision trees from scratch
"""

from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from impurities import GiniImpurity


class DecisionTree:

    def __init__(self, data, categorical_columns):

        """
        Assumption:
        First column has labels.

        Parameters:
        :param data: Whole data with labels in the first column.
        :param categorical_columns: Name of columns having categorical values.
        ________________________________________________________________________________________________________________

        Instance variables:
        columns: contains names of all the input column fields
        indices: dictionary having indices for categorical and numeric data
        train_x, test_x: training and testing input data for our decision tree
        train_y, test_y: training and testing output label for out model
        model: model for decision tree saved as dictionary

        """
        self.columns = data.columns[1:]
        data = data.values

        self.indices = {}
        self._set_indices(categorical_columns, data.shape[1]-1)

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=12)

        self.train_x = train_data[:, 1:]
        self.train_y = train_data[:, 0]

        self.test_x = test_data[:, 1:]
        self.test_y = test_data[:, 0]

        self.model = dict()

    def _set_indices(self, categorical_columns, no_of_features):
        """
        Sets indices for categorical and numeric values.
        :param categorical_columns: Name of column fields which have categorical values.
        :param no_of_features: No of total features
        """
        self.indices['categorical'] = [i for (i, j) in enumerate(self.columns) if j in categorical_columns]
        self.indices['numeric'] = list(set(range(no_of_features)) - set(self.indices['categorical']))

    @staticmethod
    def _get_gini_impurity(sett):
        """
        Returns Gini impurity given the set of data(sett).
        Note: sett would not be exact set(containing unique values).
        """
        return GiniImpurity(sett).get_impurity()

    @staticmethod
    def _get_column_data(x, categorical_index):
        """
        Sets data related to the specific column specified by categorical_index to the dictionary column_data and
        returns the dictionary.
        """
        column_data = dict()
        column_data['series'] = pd.Series(x[:, categorical_index])
        column_data['clean_series'] = column_data['series'][~column_data['series'].isnull()]
        column_data['counter'] = Counter(column_data['clean_series'])
        column_data['set'] = set(column_data['clean_series'])
        return column_data

    def _get_best_category_index(self, x, y):
        """
        Returns the best category to split the data on the basis of lowest value of impurity.
        """
        best_category_index = None

        # Setting impurity to highest value in initial (i.e 1).
        impurity = 1
        for categorical_index in self.indices['categorical']:
            column_data = self._get_column_data(x, categorical_index)
            clean_column_series, column_counter, column_set = \
                (column_data['clean_series'], column_data['counter'], column_data['set'])

            # Impurity is cumulative in the iteration so set to 0 in initial.
            impurity_t = 0
            for x_label in column_set:
                sub_labels = y[
                    x[:, categorical_index] == x_label
                    ]
                impurity_t += self._get_gini_impurity(sub_labels) * column_counter[x_label] / clean_column_series.shape[0]

            if impurity_t < impurity:
                impurity = impurity_t
                best_category_index = categorical_index

        return best_category_index, impurity

    def _get_best_numeric_index(self, x, y):
        """
        Returns the best numeric category to split the data on the basis of lowest value of impurity.
        """
        best_numeric_index = None
        best_separator = None

        # Setting impurity to highest value in initial (i.e 1).
        impurity = 1

        for numeric_index in self.indices['numeric']:
            column_data = self._get_column_data(x, numeric_index)
            clean_column_series, column_counter, column_set = \
                (column_data['clean_series'].reset_index(drop=True), column_data['counter'], column_data['set'])

            separator_points = [
                (clean_column_series[i] + clean_column_series[i+1])/2 for i in range(len(clean_column_series)-1)
            ]

            impurity_individual = 1
            best_separator_individual = None
            # Impurity is set to maximum value i.e 1 in initial.
            for separator in separator_points:
                sub_label_lesser = y[
                    x[:, numeric_index] < separator
                    ]
                sub_label_greater = y[
                    x[:, numeric_index] >= separator
                    ]
                impurity_t = self._get_gini_impurity(sub_label_lesser) * sub_label_lesser.shape[0] / clean_column_series.shape[0] + \
                            self._get_gini_impurity(sub_label_greater) * sub_label_greater.shape[0] / clean_column_series.shape[0]
                if impurity_t < impurity_individual:
                    impurity_individual = impurity_t
                    best_separator_individual = separator

            if impurity_individual < impurity:
                best_numeric_index = numeric_index
                impurity = impurity_individual
                best_separator = best_separator_individual

        return best_numeric_index, impurity, best_separator

    def train(self, x, y):
        """
        Given the labelled data with input x and output y, train the model recursively.
        ________________________________________________________________________________________________________________

        Base Condition:
        a) All the labels are same i.e. n(set(y)) = 1.
        b) There is no any feature to do further comparison.
        ________________________________________________________________________________________________________________

        Model:
        Model is saved as Python dictionary by using the same recursive traversal.

        """

        if len(set(y)) == 1:
            return y[0]

        if len(self.indices['categorical']) == 0:
            return Counter(y).most_common(1)[0][0]

        # Get best category index to do the comparison/split.
        best_category_index, categorical_impurity = self._get_best_category_index(x, y)

        # Get best numeric index and other details to do the comparison/split.
        best_numeric_index, numeric_impurity, best_separator = self._get_best_numeric_index(x, y)

        # Compare between best categorical index and numerical index and choose the lowest.
        best_index = best_category_index if categorical_impurity < numeric_impurity else best_numeric_index

        # Remove the index from the list of categorical/numeric indices to do splitting/comparison.
        if best_index == best_category_index:
            self.indices['categorical'].remove(best_index)

            # Retrieve data from the column pertaining to best_category_index.
            column_data = self._get_column_data(x, best_index)
            column_series, column_set = column_data['series'], column_data['set']

            # Construct local(or global) dictionary storing the model for the decision tree.
            model = {self.columns[best_index]: dict()}

            for x_label in column_set:
                train_x_t = x[column_series == x_label, :]
                train_y_t = y[column_series == x_label]
                model[self.columns[best_index]][x_label] = self.train(train_x_t, train_y_t)

            # using default key to account for missing values in input data to be predicted
            # the value having highest mode is taken as default value
            model[self.columns[best_index]]['default'] = Counter(self.train_x[:, best_index]).most_common(1)[0][0]

            # Append the removed categorical index
            self.indices['categorical'].append(best_index)

        else:
            self.indices['numeric'].remove(best_index)

            # Retrieve data from the column pertaining to best_category_index.
            column_data = self._get_column_data(x, best_index)
            column_series, column_set = column_data['series'], column_data['set']

            # Construct local(or global) dictionary storing the model for the decision tree.
            model = {self.columns[best_index]: dict()}

            # For greater than or equal to case
            train_x_t = x[column_series >= best_separator, :]
            train_y_t = y[column_series >= best_separator]
            model[self.columns[best_index]]['>=' + str(best_separator)] = self.train(train_x_t, train_y_t)

            # For less than case
            train_x_t = x[column_series < best_separator, :]
            train_y_t = y[column_series < best_separator]
            model[self.columns[best_index]]['< ' + str(best_separator)] = self.train(train_x_t, train_y_t)

            # using default key to account for missing values in input data to be predicted
            # the value having highest mode is taken as default value
            if sum(column_series >= best_separator) >= sum(column_series < best_separator):
                default_value = '>=' + str(best_separator)
            else:
                default_value = '< ' + str(best_separator)

            model[self.columns[best_index]]['default'] = default_value

            # Append the removed numeric index
            self.indices['numeric'].append(best_index)

        return model

    def build_model(self):
        """
        Takes training data and builds model as Python Dictionary and saves in model instance variable.
        """
        self.model = self.train(self.train_x, self.train_y)

    def predict(self, input_data, model):
        """
        Predicts and returns the label.
        :param input_data: Input dictionary of input variables.
        :param model: Python dictionary model of our decision tree.
        :return: Predicted label.
        """
        if type(model) != dict:
            return model

        indicator = list(set(input_data.keys()) & set(model.keys()))

        # Using default value if nan(not a number) i.e missing value is encountered.
        if str(input_data[indicator[0]]) == 'nan':
            input_data[indicator[0]] = model[indicator[0]]['default']

        if indicator[0] in [self.columns[i] for i in self.indices['categorical']]:

            # if the indicator is categorical just do dictionary lookup
            return self.predict(input_data, model[indicator[0]][input_data[indicator[0]]])
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
                return self.predict(input_data, model[indicator[0]]['>=' + str(comparator)])
            else:
                return self.predict(input_data, model[indicator[0]]['< ' + str(comparator)])

    def evaluate(self):
        # create tuples of dictionaries from test data(which we splitted in constructor)
        test_tuples = pd.DataFrame(self.test_x, columns=self.columns).T.to_dict().items()
        test_dict_list = [j for (i, j) in test_tuples]

        # predict the y values and save in an array
        y_pred = np.array([self.predict(i, self.model) for i in test_dict_list])
        print('Accuracy on test data:', 100 * sum((y_pred == self.test_y)) / y_pred.shape[0])


def main():

    # read data from CSV file
    data = pd.read_csv('data/preprocessed/train.csv')

    # create DecisionTree model and train using the loaded data
    dt = DecisionTree(data, categorical_columns=['Pclass', 'Sex', 'Embarked'])
    dt.build_model()

    # evaluate the model performance on 20% test data
    dt.evaluate()


if __name__ == "__main__":
    main()

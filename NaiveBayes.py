from Preprocessing import discretization, categoricalToNumeric, count_conditional_att, count_att, validator
from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self, train_file, test_file, bins=None, discretization_mode=None):
        validator(train_data=train_file, test_data=test_file, bins=bins, discretization_mode=discretization_mode)

        self.X_y_train = train_file
        self.X_y_test = test_file
        self.y_train_col_name = train_file.iloc[:, -1].name
        self.class_probabilities = []
        self.X_train_col_names = []
        self.y_prediction = []
        self.probabilities = {}
        self.discretization_mode = discretization_mode
        self.bins = int(bins)
        self.score = 0

    def loadData(self):
        """
        Initialize train and test data, drop all NaN values, make discretization on each of the Numeric columns
        :return: train and test DataFrames
        """
        for column in self.X_y_train:
            if isinstance(self.X_y_train[column].iloc[1], str):
                most_common_value = self.X_y_train[column].value_counts().idxmax()
                self.X_y_train[column] = self.X_y_train[column].fillna(most_common_value)
            else:
                column_mean = self.X_y_train[column].mean()
                self.X_y_train[column] = self.X_y_train[column].fillna(int(column_mean))

        for column in self.X_y_test:
            if isinstance(self.X_y_test[column].iloc[1], str):
                most_common_value = self.X_y_test[column].value_counts().idxmax()
                self.X_y_test[column] = self.X_y_test[column].fillna(most_common_value)
            else:
                column_mean = self.X_y_test[column].mean()
                self.X_y_test[column] = self.X_y_test[column].fillna(int(column_mean))

        self.X_y_train.reset_index(drop=True, inplace=True)
        self.X_y_test.reset_index(drop=True, inplace=True)

        for attribute in self.X_y_train[self.X_y_train.columns[-1]].unique():
            self.X_train_col_names.append(attribute)

        print("Initializing and Discretizing Data...")
        if self.bins is None or self.bins != 0:
            for column in self.X_y_train:
                if not isinstance(self.X_y_train[column][1], str) and column != self.y_train_col_name:
                    discretization(dataset=self.X_y_train, column=column, bins=self.bins,mode=self.discretization_mode)

            for column in self.X_y_test:
                if not isinstance(self.X_y_test[column][1], str) and column != self.y_train_col_name:
                    discretization(dataset=self.X_y_test, column=column, bins=self.bins,mode=self.discretization_mode)
            print("Discretization Completed!")

        for column in self.X_y_train:
            self.probabilities[column] = self.X_y_train[column].unique()

        print("Initialized Data Completed!")

    def train(self):
        """
        save all of the conditional probabilities in the dictionary for each of the values 'Age':['A': 0.2685]
        :return: yes and no dictionaries that contain all of the conditional probabilities with 'class' [yes/no] respectively
        """
        print('Training Started...')

        # Calculate Probabilities for each unique (class column) value
        no_dict = {}
        yes_dict = {}
        i = 0

        for column, array in self.probabilities.items():
            print('\rCalculating: {0}%'.format(round(i / len(self.probabilities.items()) * 100, ndigits=3)), end='')

            for value in array:
                if column.lower() != self.y_train_col_name:  # if (column) is not last classification column
                    no_dict[(column, value)] = count_conditional_att(self.X_y_train, column, value,
                                                                     self.y_train_col_name,
                                                                     self.X_train_col_names[0])

                    yes_dict[(column, value)] = count_conditional_att(self.X_y_train, column, value,
                                                                      self.y_train_col_name,
                                                                      self.X_train_col_names[1])
            i += 1
        print('\r', end='')
        self.class_probabilities.append(no_dict)
        self.class_probabilities.append(yes_dict)

    def naive_bayes_classifier(self, yes_dict, no_dict, row):
        """
        Get Test dataset, row in the test dataset to be classified, yes/no dictionaries
        :param yes_dict: conditional probabilities that contain calculation with [class: 'yes']
        :param no_dict: conditional probabilities that contain calculation with [class: 'no']
        :param row: row of data to be classified
        :return: classified row with two options: yes/no
        """
        prod_prob_y = 1
        prod_prob_n = 1
        for column in self.X_y_train:
            if column != self.y_train_col_name:
                try:
                    prod_prob_y *= yes_dict[(column, self.X_y_train[column][row])]
                except KeyError:
                    prod_prob_y *= 1
                try:
                    prod_prob_n *= no_dict[(column, self.X_y_train[column][row])]
                except KeyError:
                    prod_prob_n *= 1

        p_class_n = count_att(self.X_y_train, self.y_train_col_name, self.X_train_col_names[0])
        p_class_y = count_att(self.X_y_train, self.y_train_col_name, self.X_train_col_names[1])

        if prod_prob_y * p_class_y > prod_prob_n * p_class_n:
            return self.X_train_col_names[1]
        else:
            return self.X_train_col_names[0]

    def test(self, y_dict, n_dict):
        """
        Calculate the success rate of (naive_bayes_classifier) and return the amount of correct predictions
        :param y_dict: conditional probabilities that contain calculation with [class: 'yes']
        :param n_dict: conditional probabilities that contain calculation with [class: 'no']
        :return: amount of correct predictions
        """
        print('Testing Started...')
        for row in range(len(self.X_y_test)):
            print('\rCalculating: {0}%'.format(round(row / len(self.X_y_test) * 100, ndigits=3)), end='')

            answer = self.naive_bayes_classifier(yes_dict=y_dict, no_dict=n_dict, row=row)
            self.y_prediction.append(answer)
            real_answer = (self.X_y_test.iloc[row][-1])
            if answer == real_answer:
                self.score += 1
        print('\r', end='')
        print("Total correct was: {0}/{1} | %{2}".format(self.score, len(self.X_y_test),
                                                         round((self.score / len(self.X_y_test)) * 100, ndigits=3)))

    def run(self):
        # Load Information
        self.loadData()
        self.train()
        self.test(self.class_probabilities[1], self.class_probabilities[0])


class NaiveBayes_SKLearn:
    def __init__(self, train_file, test_file):
        validator(train_data=train_file, test_data=test_file)
        self.model = GaussianNB()
        self.y_train = train_file.iloc[:, -1]
        self.X_train = train_file.drop(columns=[train_file.iloc[:, -1].name], axis=0)
        self.y_test = test_file.iloc[:, -1]
        self.X_test = test_file.drop(columns=[test_file.iloc[:, -1].name], axis=0)
        self.y_prediction = []
        self.score = 0

    def run(self):
        # Preprocess data
        print('Converting categorical data to numeric data...')
        categoricalToNumeric(self.X_train)
        categoricalToNumeric(self.y_train)
        categoricalToNumeric(self.X_test)
        categoricalToNumeric(self.y_test)

        self.X_train.dropna(inplace=True)
        self.y_train.dropna(inplace=True)
        self.X_test.dropna(inplace=True)
        self.y_test.dropna(inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)

        # Train Model
        print('Training Model...')
        self.model.fit(self.X_train, self.y_train)

        # Test Model
        print('Testing Model...')
        for row in range(len(self.X_test)):
            print('\rCalculating: {0}%'.format(round(row / len(self.X_test) * 100, ndigits=3)), end='')
            prediction = self.model.predict([self.X_test.iloc[row]])
            self.y_prediction.append(prediction)
            if prediction == self.y_test.iloc[row]:
                self.score += 1

        print('\r', end='')
        print("Total correct was: {0}/{1} | %{2}".format(self.score, len(self.y_test),
                                                         round((self.score / len(self.y_test)) * 100, ndigits=3)))

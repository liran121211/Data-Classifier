import numpy as np
import pandas as pd

from Preprocessing import categoricalToNumeric


def pointsDistance(p1, p2):
    """
    Calculate distance between 2 points.
    :param p1: first point
    :param p2: second point
    :return: distance between 2 points
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))


class KMeans:
    def __init__(self, k_means=5, max_iterations=100, random_state=30):
        self.dataset = None
        self.num_rows = None
        self.num_features = None
        self.train = None
        self.class_ = None
        self.result = None
        self.k = k_means
        self.max_iterations = max_iterations
        self.clusters = [[] for i in range(self.k)]  # Initialize K lists inside a list
        self.centroids = []  # array of centroids

        np.random.seed(random_state)  # random for initial points.

    def prediction(self, dataset):
        """
        Classify all data points.
        :param dataset: NumPy dataset
        :return: labeled clusters
        """
        print("Clustering data has started...")
        self.dataset = dataset
        self.num_rows, self.num_features = dataset.shape

        # Initialize a value between min-max size of dataset
        random_indexes = np.random.choice(self.num_rows, self.k, replace=False)
        self.centroids = [self.dataset[index] for index in random_indexes]  # size (k) array

        # Optimize clusters (max_iterations) times
        for i in range(self.max_iterations):
            self.clusters = self.createClusters(self.centroids)

            # Calculate new centroids from the clusters
            previous_centroids = self.centroids
            self.centroids = self.getCentroids(self.clusters)

            # Check if clusters have changed
            if self.converged(previous_centroids, self.centroids):
                break

        # Classify points as the index of their clusters
        self.result = self.labelClusters(self.clusters)

    def labelClusters(self, clusters_):
        """
        Create an array with labeled points per cluster.
        :param clusters_: clusters
        :return: the labels
        """
        labels = np.empty(self.num_rows)
        for cluster_index, cluster in enumerate(clusters_):
            for index in cluster:
                labels[index] = cluster_index
        return labels

    def createClusters(self, centroids):
        """
        Classify the points to the closest centroids in order to create the clusters
        :param centroids: array of centroids
        :return:  classified points by cluster
        """
        clusters_ = [[] for i in range(self.k)]  # create empty (k) arrays inside array
        for index, row in enumerate(self.dataset):
            centroid_index = self.closestCentroid(row, centroids)  # returns index of the closest centroid in the array
            clusters_[centroid_index].append(
                index)  # fill the clusters arrays with the index of the closets points to it
        return clusters_

    def closestCentroid(self, point, centroids):
        """
        Find the closest centroid to the given point
        :param point: D dimension point.
        :param centroids: array of centroids
        :return: return the index of the closest centroid to the point.
        """
        # distance of the current sample to each centroid
        distances = [pointsDistance(point, centroid) for centroid in
                     centroids]  # find the closest point from the centroid for each point
        closest_index = np.argmin(distances)  # get the most minimal index of the centroids from (distances)
        return closest_index

    def getCentroids(self, clusters_):
        """
        Calculate new centroids for the updated clusters.
        :param clusters_: array of clusters
        :return: new array of updated cluster coordinates
        """
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.k, self.num_features))  # nullify a new centroids array to find new centroids
        for cluster_index, cluster in enumerate(clusters_):
            cluster_mean = np.mean(self.dataset[cluster], axis=0)  # find new mean for each cluster
            centroids[cluster_index] = cluster_mean
        return centroids

    def converged(self, centroids_old, centroids):
        """
        Check if centroids coordinates is changed.
        :param centroids_old: old centroid points
        :param centroids:  new centroid points
        :return: true if the same coordinates, false if not .
        """
        # calculate the distances between old and new centroids
        distances = [pointsDistance(centroids_old[j], centroids[j]) for j in range(self.k)]
        return sum(distances) == 0

    def loadData(self, file, columns_=None):
        """
        Load and preprocess the datasets.
        dataset: Pandas dataframe file
        :return: processed datasets.
        """
        # load data files
        self.dataset = file

        # clean missing data
        self.dataset.dropna(inplace=True)
        self.dataset.reset_index(drop=True, inplace=True)

        # convert to numeric values
        categoricalToNumeric(self.dataset)

        # final preprocessing
        class_col = self.dataset[self.dataset.columns[-1]].copy()

        if columns_ is None:
            train = self.dataset
        else:
            train = self.dataset[columns_]

        # Scale dataset with zScore method
        for col in self.dataset:
            self.dataset[col] = (self.dataset[col] - self.dataset[col].mean()) / self.dataset[col].std(ddof=0)

        # Savw processed data to object
        self.train = train.to_numpy()
        self.class_ = class_col

    def score(self):
        """
        Test the accuracy of the clustered data
        :return: Score 0-100% of correct guess
        """
        print("Calculating score...")
        match = 0
        for i in range(len(self.result)):
            if self.result[i] == self.class_[i]:
                match += 1

        print("Success Rate: {0}".format((match / len(self.result)) * 100))


def run():
    file = pd.read_csv('train.csv', delimiter=',')
    k_means = KMeans(k_means=2, max_iterations=150, random_state=49)
    k_means.loadData(file)
    k_means.prediction(k_means.train)
    k_means.score()


# Run Model
run()

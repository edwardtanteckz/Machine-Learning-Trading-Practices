import numpy as np


class RTLearner(object):
    """
    This is a Random Tree Learner (RTLearner). You will need to properly implement this class as necessary.

    Parameters
        leaf_size (int)  - Is the number of samples required for a potential split
        verbose (bool)   - If “verbose” is True, your code can print out information for debugging.
                           If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    """

    _TREE_LEAF = -1  # constant to indicate a leaf node in the tree
    _TREE_FEATURE = 0  # index for feature in the tree array
    _TREE_SPLIT = 1  # index for split value in the tree array
    _TREE_LEFT = 2  # index for left child in the tree array
    _TREE_RIGHT = 3  # index for right child in the tree array

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "etan74"  # replace tb34 with your Georgia Tech username

    def study_group(self):
        return self.author()
    

    def _build_tree(self, data_x, data_y):
        n_samples, _ = data_x.shape

        if n_samples <= self.leaf_size:
            leaf_value = np.mean(data_y)
            return np.array([[self._TREE_LEAF, leaf_value, np.nan, np.nan]])

        if np.all(data_y == data_y[0]):
            leaf_value = data_y[0]
            return np.array([[self._TREE_LEAF, leaf_value, np.nan, np.nan]])

        feature_index = np.random.randint(0, data_x.shape[1])
        idx_a, idx_b = np.random.choice(n_samples, 2, replace=False)
        split_value = (data_x[idx_a, feature_index] + data_x[idx_b, feature_index]) / 2.0

        left_mask = data_x[:, feature_index] <= split_value
        right_mask = data_x[:, feature_index] > split_value

        if not np.any(left_mask) or not np.any(right_mask):
            leaf_value = np.mean(data_y)
            return np.array([[self._TREE_LEAF, leaf_value, np.nan, np.nan]])
        
        left_tree = self._build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self._build_tree(data_x[right_mask], data_y[right_mask])

        tree = np.vstack([
            [feature_index, split_value, 1, left_tree.shape[0] + 1],
            left_tree,
            right_tree
        ])
        return tree
    
    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        self.tree = self._build_tree(data_x, data_y)
        if(self.verbose):
            print(f"Decision Tree built successfully: {self.tree.shape}")

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        predictions = np.zeros(points.shape[0])
        for i, point in enumerate(points):
            node = 0
            while self.tree[node, self._TREE_FEATURE] != self._TREE_LEAF:
                feature_index = int(self.tree[node, self._TREE_FEATURE])
                split_value = self.tree[node, self._TREE_SPLIT]
                if point[feature_index] <= split_value:
                    node += int(self.tree[node, self._TREE_LEFT])
                else:
                    node += int(self.tree[node, self._TREE_RIGHT])
            predictions[i] = self.tree[node, self._TREE_SPLIT]
        return predictions

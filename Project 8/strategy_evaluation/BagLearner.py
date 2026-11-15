import numpy as np


class BagLearner(object):
    """
    This is a Bootstrap Aggregation Learner (BagLearner). You will need to properly implement this class as necessary.

    Parameters
        learner (learner) - Points to any arbitrary learner class that will be used in the BagLearner.
        kwargs            - Keyword arguments that are passed on to the learner’s constructor and they can vary according to the learner
                          - NOTE: kwargs is shown in the API as it would be used, but it is NOT required to be a default argument in the same
                            way as bags, boost, or verbose.
        bags (int)        - The number of learners you should train using Bootstrap Aggregation. 
                            If boost is true, then you should implement boosting (optional implementation).
        verbose (bool)    - If “verbose” is True, your code can print out information for debugging.
                            If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    """
    def __init__(self, learner, kwargs = {}, bags = 20, boost = False, verbose = False):
        """
        Constructor method
        """
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = [learner(**kwargs) for _ in range(bags)]

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "etan74"  # replace tb34 with your Georgia Tech username

    def study_group(self):
        return self.author()

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        n_samples = data_x.shape[0]
        for learner in self.learners:
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            learner.add_evidence(data_x[indices], data_y[indices])

        if self.verbose:
            print(f"Trained {self.bags} learners with bootstrap samples: {n_samples}.")

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        predictions = np.array([learner.query(points) for learner in self.learners])
        return np.mean(predictions, axis=0)

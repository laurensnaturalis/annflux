import unittest

from annflux.training.annflux.quick import make_predictions


class TestKnnPredictions(unittest.TestCase):
    def setUp(self):
        import pickle

        # Specify the file name from which you want to load the pickled objects
        pickle_file = "/home/lhogeweg/Documents/annflux_ln/src/annflux/ui/basic/make_predictions_input.pkl"  # TODO

        # Load the pickled objects from the file
        with open(pickle_file, "rb") as file:
            loaded_objects = pickle.load(file)

        # Extract the objects from the dictionary
        self.annotations = loaded_objects["annotations"]
        self.data = loaded_objects["data"]
        self.indices_ = loaded_objects["indices_"]
        self.distances_ = loaded_objects["distances_"]
        self.label_array = loaded_objects["label_array"]
        self.near_labeled_indices_ = loaded_objects["near_labeled_indices_"]
        self.test_indices = loaded_objects["test_indices"]
        self.knn_rank_exponent = loaded_objects["knn_rank_exponent"]

        print("Objects have been loaded from the pickle file.")

    def test_speed(self):
        print("bla")
        predicted_test, true_test, _  = make_predictions(
            self.annotations,
            self.data,
            self.indices_,
            self.distances_,
            self.label_array,
            self.near_labeled_indices_,
            self.test_indices,
            knn_rank_exponent=self.knn_rank_exponent,
            # skip_first=True
        )

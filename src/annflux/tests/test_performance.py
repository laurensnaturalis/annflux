from annflux.performance.basic import compute_performance
import pickle
import unittest


class TestPerformance(unittest.TestCase):
    def test_speed(self):
        with open(
            "/home/lhogeweg/Documents/annflux_ln/src/annflux/ui/basic/compute_performance_test.pkl",
            "rb",
        ) as f:
            loaded_results = pickle.load(f)
        predicted_test = loaded_results["predicted_test"]
        true_test = loaded_results["true_test"]
        annotations = loaded_results["annotations"]
        data = loaded_results["data"]

        compute_performance(
            predicted_test,
            true_test,
            annotations,
            data,
            100,
            performance_graph_path="tmp.json",
            detailed_performance_path="tmp.csv",
        )

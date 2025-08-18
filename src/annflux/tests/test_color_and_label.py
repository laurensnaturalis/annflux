import json
import unittest

import pandas

from annflux.tools.data import color_and_label


class TestColorAndLabel(unittest.TestCase):
    def test_color_and_label(self):
        multilabel_to_color, class_to_count = color_and_label(
            pandas.read_csv(
                "/mnt/big/Projects/diopsis_annotation_2/annflux/annflux.csv"
            ),
            json.load(
                open("/mnt/big/Projects/diopsis_annotation_2/annflux/labels.json")
            ),
            json.load(
                open("/mnt/big/Projects/diopsis_annotation_2/annflux/label_defs.json")
            )["labels"],
        )

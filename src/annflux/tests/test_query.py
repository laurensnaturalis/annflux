import unittest

import pandas

from annflux.tools.io import sql_to_pandas_query


class TestQuery(unittest.TestCase):
    def setUp(self):
        pass

    def test_query(self):
        sql_to_pandas_query(
            "(Pap IN row.label_predicted OR Pap IN row.label_true)",
            pandas.DataFrame(
                data={"label_predicted": ["Pap", "Bap"], "label_true": ["Bap", "Pap"]}
            ),
        )

    def test_not_query(self):
        sql_to_pandas_query(
            "(row.label_predicted != 'Empty')",
            pandas.DataFrame(
                data={"label_predicted": ["Pap", "Bap"], "label_true": ["Bap", "Pap"]}
            ),
        )

    def test_not_in_query(self):
        sql_to_pandas_query(
            "(Empty NOT IN row.label_predicted)",
            pandas.DataFrame(
                data={"label_predicted": ["Pap", "Bap"], "label_true": ["Bap", "Pap"]}
            ),
        )

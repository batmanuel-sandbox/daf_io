from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
import os
from sqlalchemy import create_engine
import lsst.utils.tests
from lsst.afw import table as afw_table

from lsst.daf.io.table.sql import get_schema, to_sql, read_sql

SQLITE_LOCATION = os.path.dirname(__file__) + "/test.db"


class TableIoTestCase(lsst.utils.tests.TestCase):

    def tearDown(self):
        if os.path.exists(SQLITE_LOCATION):
            os.remove(SQLITE_LOCATION)

    def test_write_read(self):
        """Test that a Schema can be read from a FITS file

        Per DM-8211.
        """
        cat_expected = _get_afw_table()
        engine = create_engine("sqlite:///" + SQLITE_LOCATION)

        # Test writing first
        # FIXME: Hopefully tables implement __eq__ in the future
        to_sql(cat_expected, "testname", engine)

        # Test reading back with raw object
        rows = engine.execute("select a, b from testname")
        self._compare_table(cat_expected, rows)

        # Test reading back with different SQL
        cat = read_sql("SELECT * from testname", engine)
        self._compare_table(cat_expected, cat)

        # Test reading back again with table name only
        cat = read_sql("testname", engine)
        self._compare_table(cat_expected, cat)

    def test_dtype_override(self):
        engine = create_engine("sqlite:///" + SQLITE_LOCATION)
        cat = _get_afw_table()
        to_sql(cat, "testname", engine)
        engine = create_engine("sqlite:///" + SQLITE_LOCATION)
        cat = read_sql("SELECT * from testname", engine,
                       column_dtypes={"a": np.int32})
        self.assertEquals(np.int32, cat.schema['a'].asField().dtype)

        cat = read_sql("SELECT * from testname", engine,
                       column_dtypes={"a": np.float32})
        self.assertEquals(np.float32, cat.schema['a'].asField().dtype)

    def test_get_schema(self):
        cat = _get_afw_table()
        get_schema(cat, "testname")

        engine = create_engine("sqlite:///" + SQLITE_LOCATION)
        get_schema(cat, "testname", con=engine)

        engine = create_engine("mysql://localhost")
        get_schema(cat, "testname", con=engine)

    def _compare_table(self, afw_cat, rows):
        columns = [str(i.field.getName()) for i in afw_cat.schema]
        for cat1_row, cat2_row in zip(afw_cat, rows):
            for column in columns:
                self.assertEquals(cat1_row[column], cat2_row[column])


def _get_afw_table():
    schema = afw_table.Schema()
    aa = schema.addField("a", type=np.int64, doc="a")
    bb = schema.addField("b", type=np.float64, doc="b")
    cat = afw_table.BaseCatalog(schema)
    row = cat.addNew()
    row.set(aa, 12345)
    row.set(bb, 1.2345)
    row = cat.addNew()
    row.set(aa, 4321)
    row.set(bb, 4.123)
    return cat


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

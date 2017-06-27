from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

from sqlalchemy import create_engine
import lsst.utils.tests
import lsst.afw.table

from lsst.afw.table.io.sql import get_schema

class TableIoTestCase(lsst.utils.tests.TestCase):


    def testSchemaReading(self):
        """Test that a Schema can be read from a FITS file

        Per DM-8211.
        """
        schema = lsst.afw.table.Schema()
        aa = schema.addField("a", type=np.int64, doc="a")
        bb = schema.addField("b", type=np.float64, doc="b")
        schema.getAliasMap().set("c", "a")
        schema.getAliasMap().set("d", "b")
        cat = lsst.afw.table.BaseCatalog(schema)
        row = cat.addNew()
        row.set(aa, 12345)
        row.set(bb, 1.2345)

        engine = create_engine("mysql://localhost")
        print(get_schema(cat, engine))


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

# -*- coding: utf-8 -*-
"""
Collection of query wrappers / abstractions to both facilitate data
retrieval and to reduce dependency on DB-specific API.
"""

from __future__ import print_function, division
from builtins import str, bytes, zip, range

import numpy as np
import sqlalchemy
import struct

from lsst.afw import table as afw_table
from sqlalchemy import select, Table, Column, Numeric
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import MetaData, CreateTable
from sqlalchemy.types import (SmallInteger, Integer, BigInteger,
                              Float, String, Boolean, Text)
from sqlalchemy.types import to_instance, TypeEngine
from contextlib import contextmanager


MAX_ARRAY_SIZE = 65535


class DatabaseError(IOError):
    pass


def read_sql_table(table_name, con, schema=None, column_dtypes=None,
                   coerce_float=True, parse_dates=None, columns=None,
                   chunksize=None):
    """Read SQL database table into an afw table.

    Given a table name and an SQLAlchemy connectable, returns an Table.
    This function does not support DBAPI connections.

    Parameters
    ----------
    table_name : string
        Name of SQL table in database
    con : SQLAlchemy connectable (or database string URI)
    schema : string, default: None
        Name of SQL schema in database to query (if database flavor
        supports this). If None, use default schema (default).
    column_dtypes : dict, default: None
        Dict of ``{column_name: dtype}`` where dtype is the target
        dtype of the column. This is used to harmonize types and downcast
        from doubles and longs.
    coerce_float : boolean, default True
        Attempt to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Can result in loss of Precision.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps
    columns : list, default: None
        List of column names to select from sql table
    chunksize : int, default: None
        If specified, return an iterator where ``chunksize`` is the number
        of rows to include in each chunk.

    Returns
    -------
    afw table

    Notes
    -----
    Any datetime values with time zone information will be converted to UTC

    See also
    --------
    read_sql_query : Read SQL query into a afw table.
    read_sql

    """

    con = _engine_builder(con)
    meta = MetaData(con, schema=schema)
    try:
        meta.reflect(only=[table_name], views=True)
    except sqlalchemy.exc.InvalidRequestError:
        raise ValueError("Table %s not found" % table_name)

    sql_io = SQLDatabase(con, meta=meta)
    table = sql_io.read_table(
        table_name, column_dtypes=column_dtypes, coerce_float=coerce_float,
        parse_dates=parse_dates, columns=columns, chunksize=chunksize)

    if table is not None:
        return table
    else:
        raise ValueError("Table %s not found" % table_name, con)


def read_sql_query(sql, con, column_dtypes=None, coerce_float=True,
                   params=None, parse_dates=None, chunksize=None):
    """Read SQL query into an afw table.

    Returns an afw table corresponding to the result set of the query
    string. Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    Parameters
    ----------
    sql : string SQL query or SQLAlchemy Selectable (select or text object)
        to be executed.
    con : SQLAlchemy connectable(engine/connection) or database string URI
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
    column_dtypes : dict, default: None
        Dict of ``{column_name: dtype}`` where dtype is the target
        dtype of the column. This is used to harmonize types and downcast
        from doubles and longs.
    coerce_float : boolean, default True
        Attempt to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}
    parse_dates : list or dict, default: None
        - List of column names to parse as dates
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps
    chunksize : int, default: None
        If specified, return an iterator where ``chunksize`` is the number
        of rows to include in each chunk.

    Returns
    -------
    afw table

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC

    See also
    --------
    read_sql_table : Read SQL database table into a afw table
    read_sql

    """
    sql_io = _sql_io_builder(con)
    return sql_io.read_sql(
        sql, params=params, column_dtypes=column_dtypes,
        coerce_float=coerce_float, parse_dates=parse_dates, chunksize=chunksize)


def read_sql(sql, con, column_dtypes=None, coerce_float=True, params=None,
             parse_dates=None, columns=None, chunksize=None):
    """
    Read SQL query or database table into a afw table.

    Parameters
    ----------
    sql : string SQL query or SQLAlchemy Selectable (select or text object)
        to be executed, or database table name.
    con : SQLAlchemy connectable(engine/connection) or database string URI
        or DBAPI2 connection (fallback mode)
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
    column_dtypes : dict, default: None
        Dict of ``{column_name: dtype}`` where dtype is the target
        dtype of the column. This is used to harmonize types and downcast
        from doubles and longs.
    coerce_float : boolean, default True
        Attempt to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}
    parse_dates : list or dict, default: None
        - List of column names to parse as dates
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps
    columns : list, default: None
        List of column names to select from sql table (only used when reading
        a table).
    chunksize : int, default: None
        If specified, return an iterator where ``chunksize`` is the number
        of rows to include in each chunk.

    Returns
    -------
    afw table

    Notes
    -----
    This function is a convenience wrapper around ``read_sql_table`` and
    ``read_sql_query`` (and for backward compatibility) and will delegate
    to the specific function depending on the provided input (database
    table name or sql query).  The delegated function might have more specific
    notes about their functionality not listed here.

    See also
    --------
    read_sql_table : Read SQL database table into a afw table
    read_sql_query : Read SQL query into a afw table

    """
    sql_io = _sql_io_builder(con)

    try:
        _is_table_name = sql_io.table_exists(sql)
    except SQLAlchemyError:
        _is_table_name = False

    if _is_table_name:
        sql_io.meta.reflect(only=[sql])
        return sql_io.read_table(
            sql, column_dtypes=column_dtypes, coerce_float=coerce_float,
            parse_dates=parse_dates, columns=columns, chunksize=chunksize)
    else:
        return sql_io.read_sql(
            sql, params=params, column_dtypes=column_dtypes,
            coerce_float=coerce_float, parse_dates=parse_dates,
            chunksize=chunksize)


def to_sql(catalog, name, con, schema=None, if_exists='fail',
           chunksize=None, dtype=None):
    """
    Write records stored in a afw table to a SQL database.

    Parameters
    ----------
    catalog : afw table
    name : string
        Name of SQL table
    con : SQLAlchemy connectable(engine/connection) or database string
        URI. Using SQLAlchemy makes it possible to use any DB supported
        by that library.
    schema : string, default: None
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    if_exists : {'fail', 'replace', 'append'}, default 'fail'
        - fail: If table exists, do nothing.
        - replace: If table exists, drop it, recreate it, and insert data.
        - append: If table exists, insert data. Create if does not exist.
    chunksize : int, default: None
        If not None, then rows will be written in batches of this size at a
        time.  If None, all rows will be written at once.
    dtype : single SQLtype or dict of column name to SQL type, default: None
        Optional specifying the datatype for columns. The SQL type should
        be a SQLAlchemy type.
        If all columns are of the same type, one single value can be used.

    """
    if if_exists not in ('fail', 'replace', 'append'):
        raise ValueError("'{0}' is not valid for if_exists".format(if_exists))

    sql_io = _sql_io_builder(con, schema=schema)
    sql_io.to_sql(catalog, name, if_exists=if_exists, schema=schema,
                  chunksize=chunksize, dtype=dtype)


def table_exists(table_name, con, schema=None):
    """
    Check if the Database has named table.

    Parameters
    ----------
    table_name: string
        Name of SQL table
    con: SQLAlchemy connectable (engine/connection).
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
    schema : string, default: None
        Name of SQL schema in database to write to (if database flavor supports
        this). If None, use default schema (default).

    Returns
    -------
    boolean
    """
    sql_io = _sql_io_builder(con, schema=schema)
    return sql_io.table_exists(table_name)


# -----------------------------------------------------------------------------
# -- Helper functions

def _is_dict_like(obj):
    return hasattr(obj, '__getitem__') and hasattr(obj, 'keys')


def _is_list_like(obj):
    return hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes))


def _sql_io_builder(con, schema=None, meta=None):
    """Convenience function to return a SQLIO object"""
    con = _engine_builder(con)
    return SQLDatabase(con, schema=schema, meta=meta)


def _convert_params(sql, params):
    """convert sql and params args"""
    args = [sql]
    if params is not None:
        if _is_dict_like(params):
            args += [params]
        else:
            args += [list(params)]
    return args


def _wrap_result(data, column_names, table=None, index_col=None,
                 coerce_float=True, column_dtypes=None, parse_dates=None):
    """Wrap result set of query in a afw table """

    result_size = len(data)
    # Turn into columns first

    from pandas import lib
    data = list(lib.to_object_array_tuples(data).T)
    arrays = [lib.maybe_convert_objects(arr, try_float=True) for arr in data]

    _harmonize_columns(arrays, column_names, table, column_dtypes,
                       parse_dates)
    schema = afw_table.Schema()

    # build schema
    for i, column_name in enumerate(column_names):
        column_type = arrays[i].dtype
        schema.addField(column_name, type=column_type.type)

    catalog = afw_table.BaseCatalog(schema)

    # Preallocate rows based on first column length
    catalog.preallocate(result_size)
    for i in range(result_size):
        record = catalog.addNew()
        for column_i in range(len(column_names)):
            record.set(column_names[column_i], arrays[column_i][i])
    return catalog


def _engine_builder(con):
    """
    Returns a SQLAlchemy engine from a URI (if con is a string)
    else it just return con without modifying it
    """
    if isinstance(con, str):
        con = sqlalchemy.create_engine(con)
        return con

    return con


class SQLTable(object):
    """
    For mapping afw.table to SQL tables.
    Uses fact that table is reflected by SQLAlchemy to
    do better type conversions.
    Also holds various flags needed to avoid having to
    pass them between functions all the time.

    Users are NOT adviced to use this directly.
    """

    def __init__(self, name, sql_io_engine, catalog=None, if_exists='fail',
                 schema=None, keys=None, dtype=None):
        self.name = name
        self.sql_io = sql_io_engine
        self.catalog = catalog
        self.schema = schema
        self.if_exists = if_exists
        self.keys = keys
        self.dtype = dtype

        if catalog is not None:
            # We want to initialize based on a catalog
            self.sql_table = self._create_table_setup()
        else:
            # no data provided, read-only mode
            self.sql_table = self.sql_io.get_table(self.name, self.schema)

        if self.sql_table is None:
            raise ValueError("Could not init table '%s'" % name)

    def exists(self):
        return self.sql_io.table_exists(self.name, self.schema)

    def sql_schema(self):
        return str(CreateTable(self.sql_table).compile(self.sql_io.connectable))

    def _execute_create(self):
        # Inserting table into database, add to MetaData object
        self.sql_table = self.sql_table.tometadata(self.sql_io.meta)
        self.sql_table.create()

    def create(self):
        if self.exists():
            if self.if_exists == 'fail':
                raise ValueError("Table '%s' already exists." % self.name)
            elif self.if_exists == 'replace':
                self.sql_io.drop_table(self.name, self.schema)
                self._execute_create()
            elif self.if_exists == 'append':
                pass
            else:
                raise ValueError(
                    "'{0}' is not valid for if_exists".format(self.if_exists))
        else:
            self._execute_create()

    def insert_statement(self):
        return self.sql_table.insert()

    def insert_data(self):
        # TODO: temp.columns.extract("*"), NaN handling, DateTime?
        temp = self.catalog
        column_names = [x.field.getName() for x in temp.schema]
        ncols = len(column_names)
        data_list = [None] * ncols

        for name, i in zip(column_names, range(ncols)):
            data_list[i] = np.array(temp.columns[name], dtype=object)
        return column_names, data_list

    def _execute_insert(self, conn, keys, data_iter):
        data = []
        for row in data_iter:

            data.append(dict((k, v) for k, v in zip(keys, row)))
        conn.execute(self.insert_statement(), data)

    def insert(self, chunksize=None):
        keys, data_list = self.insert_data()

        nrows = len(self.catalog)

        if nrows == 0:
            return

        if chunksize is None:
            chunksize = nrows
        elif chunksize == 0:
            raise ValueError('chunksize argument should be non-zero')

        chunks = int(nrows / chunksize) + 1

        with self.sql_io.run_transaction() as conn:
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, nrows)
                if start_i >= end_i:
                    break
                chunk_iter = zip(*[arr[start_i:end_i] for arr in data_list])
                self._execute_insert(conn, keys, chunk_iter)

    def read(self, coerce_float=True, parse_dates=None, column_names=None,
             column_dtypes=None, chunksize=None):

        if column_names is not None and len(column_names) > 0:
            cols = [self.sql_table.c[n] for n in column_names]
            sql_select = select(cols)
        else:
            sql_select = self.sql_table.select()

        result = self.sql_io.execute(sql_select)
        column_names = result.keys()

        if chunksize is not None:
            return self._query_iterator(result, chunksize, column_names,
                                        column_dtypes=column_dtypes,
                                        coerce_float=coerce_float,
                                        parse_dates=parse_dates)
        else:
            data = result.fetchall()
            return _wrap_result(data, column_names, table=self.sql_table,
                                column_dtypes=column_dtypes,
                                coerce_float=coerce_float,
                                parse_dates=parse_dates)

    def _query_iterator(self, result, chunksize, columns, coerce_float=True,
                        column_dtypes=None, parse_dates=None):
        """Return generator through chunked result set"""

        while True:
            data = result.fetchmany(chunksize)
            if not data:
                break
            else:
                yield _wrap_result(data, columns, table=self.sql_table,
                                   column_dtypes=column_dtypes,
                                   coerce_float=coerce_float,
                                   parse_dates=parse_dates)

    def _get_column_names_and_types(self):

        column_names_and_types = []
        for schemaItem in self.catalog.schema:
            field = schemaItem.field
            name = str(field.getName())
            column_type = self._sqlalchemy_type(field)
            column_names_and_types.append((name, column_type))

        return column_names_and_types

    def _create_table_setup(self):

        column_names_and_types = self._get_column_names_and_types()

        # TODO: Do we need to support indexes?
        # columns = [Column(name, typ, index=False)
        columns = [Column(name, typ)
                   for name, typ in column_names_and_types]

        # TODO: How does afw.table deal with primary keys?
        # if self.keys is not None:
        #     if not is_list_like(self.keys):
        #         keys = [self.keys]
        #     else:
        #         keys = self.keys
        #     pkc = PrimaryKeyConstraint(*keys, name=self.name + '_pk')
        #     columns.append(pkc)
        schema = self.schema or self.sql_io.meta.schema

        # At this point, attach to new metadata, only attach to self.meta
        # once table is created.
        meta = MetaData(self.sql_io, schema=schema)

        return Table(self.name, meta, *columns, schema=schema)

    def _sqlalchemy_type(self, field):

        dtype = self.dtype or {}
        if field.getName() in dtype:
            return self.dtype[field.getName()]

        col_type = field.getTypeString()

        if col_type == "U":
            return SmallInteger
        elif col_type == "I":
            return Integer
        elif col_type == "L":
            return BigInteger
        elif col_type == "F":
            return Float(precision=23)
        elif col_type == "D":
            return Float(precision=53)
        elif col_type == "Flag":
            return Boolean
        elif col_type == "Angle":
            return Float(precision=53)
        elif col_type == "String":
            return String(length=field.getSize())
        elif col_type == "ArrayU":
            return self._get_array_type("H", field)
        elif col_type == "ArrayI":
            return self._get_array_type("i", field)
        elif col_type == "ArrayF":
            return self._get_array_type("f", field)
        elif col_type == "ArrayD":
            return self._get_array_type("d", field)
        elif col_type == 'complex':
            raise ValueError('Complex datatypes not supported')

        return Text

    def _get_array_type(self, format_char, field):
        sz = field.getSize()
        if sz == 0:
            return "BLOB NOT NULL"
        sz *= struct.calcsize("<" + format_char)
        if sz > MAX_ARRAY_SIZE:
            raise RuntimeError("Array field is too large for ingestion")
        return "BINARY({}) NOT NULL".format(sz)


def _harmonize_columns(data, column_names, table=None,
                       column_dtypes=None, parse_dates=None):
    """
    Make afw.table's column types align with the SQL table
    column types.
    Need to work around limited NA value support. Floats are always
    fine, ints must always be floats if there are Null values.
    Booleans are hard because converting bool column with None replaces
    all Nones with false. Therefore only convert bool if there are no
    NA values.
    Datetimes should already be converted to np.datetime64 if supported,
    but here we also force conversion if required
    """

    column_dtypes = column_dtypes or {}
    # TODO: handle parse_dates

    for i, col_name in enumerate(column_names):
        try:
            col_type = None
            df_col = data[i]
            # the type the afw table column should have
            if col_name in column_dtypes:
                col_type = column_dtypes[col_name]
            elif table is not None and col_name in table.columns:
                col_type = _get_dtype(table.columns.get(col_name).type)

            # TODO: DateTime, see above
            if col_type is float:
                # floats support NA, can always convert!
                data[i] = df_col.astype(col_type, copy=False)

            # TODO: Handle null types
            if col_type:
                data[i] = df_col.astype(col_type, copy=False)

        except KeyError:
            pass  # this column not in results


def _get_dtype(sqltype):
    # TODO: Handle DateTimes
    from sqlalchemy.types import (Integer, Float, Boolean)
    if isinstance(sqltype, Float):
        return float
    elif isinstance(sqltype, Integer):
        # TODO: Refine integer size.
        return np.dtype('int64')
    elif isinstance(sqltype, Boolean):
        return bool
    return object


class SQLDatabase(object):
    """
    This class enables conversion between afw table and SQL databases
    using SQLAlchemy to handle Database abstraction

    Parameters
    ----------
    engine : SQLAlchemy connectable
        Connectable to connect with the database. Using SQLAlchemy makes it
        possible to use any DB supported by that library.
    schema : string, default: None
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    meta : SQLAlchemy MetaData object, default: None
        If provided, this MetaData object is used instead of a newly
        created. This allows to specify database flavor specific
        arguments in the MetaData object.

    """

    def __init__(self, engine, schema=None, meta=None):
        self.connectable = engine
        if not meta:
            meta = MetaData(self.connectable, schema=schema)

        self.meta = meta

    @contextmanager
    def run_transaction(self):
        with self.connectable.begin() as tx:
            if hasattr(tx, 'execute'):
                yield tx
            else:
                yield self.connectable

    def execute(self, *args, **kwargs):
        """Simple passthrough to SQLAlchemy connectable"""
        return self.connectable.execute(*args, **kwargs)

    def read_table(self, table_name, column_dtypes=None, coerce_float=True,
                   parse_dates=None, columns=None, schema=None,
                   chunksize=None):
        """Read SQL database table into an afw.table.

        Parameters
        ----------
        table_name : string
            Name of SQL table in database
        column_dtypes : dict, default: None
            Dict of ``{column_name: dtype}`` where dtype is the target
            dtype of the column. This is used to harmonize types and downcast
            from doubles and longs, and override reflected types.
        coerce_float : boolean, default True
            Attempt to convert values of non-string, non-numeric objects
            (like decimal.Decimal) to floating point. This can result in
            loss of precision.
        parse_dates : list or dict, default: None
            - List of column names to parse as dates
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps
        columns : list, default: None
            List of column names to select from sql table
        schema : string, default: None
            Name of SQL schema in database to query (if database flavor
            supports this).  If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default: None
            If specified, return an iterator where ``chunksize`` is the number
            of rows to include in each chunk.

        Returns
        -------
        afw.table

        See also
        --------
        read_sql_table
        SQLDatabase.read_sql

        """
        table = SQLTable(table_name, self, schema=schema)
        return table.read(column_dtypes=column_dtypes,
                          coerce_float=coerce_float, parse_dates=parse_dates,
                          column_names=columns, chunksize=chunksize)

    @staticmethod
    def _query_iterator(result, chunksize, columns, index_col=None,
                        column_dtypes=None, coerce_float=True,
                        parse_dates=None):
        """Return generator through chunked result set"""

        while True:
            data = result.fetchmany(chunksize)
            if not data:
                break
            else:
                yield _wrap_result(data, columns, index_col=index_col,
                                   column_dtypes=column_dtypes,
                                   coerce_float=coerce_float,
                                   parse_dates=parse_dates)

    def read_sql(self, sql, index_col=None, column_dtypes=None, coerce_float=True,
                 parse_dates=None, params=None, chunksize=None):
        """Read SQL query into a afw.table.

        Parameters
        ----------
        sql : string
            SQL query to be executed
        index_col : string, optional, default: None
            Column name to use as index for the returned afw.table object.
        column_dtypes : dict, default: None
            Dict of ``{column_name: dtype}`` where dtype is the target
            dtype of the column. This is used to harmonize types and downcast
            from doubles and longs.
        coerce_float : boolean, default True
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets
        params : list, tuple or dict, optional, default: None
            List of parameters to pass to execute method.  The syntax used
            to pass parameters is database driver dependent. Check your
            database driver documentation for which of the five syntax styles,
            described in PEP 249's paramstyle, is supported.
            Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}
        parse_dates : list or dict, default: None
            - List of column names to parse as dates
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps
        chunksize : int, default: None
            If specified, return an iterator where ``chunksize`` is the number
            of rows to include in each chunk.

        Returns
        -------
        afw.table

        See also
        --------
        read_sql_table : Read SQL database table into a afw.table

        """
        args = _convert_params(sql, params)
        result = self.execute(*args)
        columns = result.keys()

        if chunksize is not None:
            return self._query_iterator(result, chunksize, columns,
                                        column_dtypes=column_dtypes,
                                        index_col=index_col,
                                        coerce_float=coerce_float,
                                        parse_dates=parse_dates)
        else:
            data = result.fetchall()
            return _wrap_result(data, columns, index_col=index_col,
                                column_dtypes=column_dtypes,
                                coerce_float=coerce_float,
                                parse_dates=parse_dates)

    def to_sql(self, catalog, name, if_exists='fail',
               schema=None, chunksize=None, dtype=None):
        """
        Write records stored in a afw table to a SQL database.

        Parameters
        ----------
        catalog : afw table
        name : string
            Name of SQL table
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            - fail: If table exists, do nothing.
            - replace: If table exists, drop it, recreate it, and insert data.
            - append: If table exists, insert data. Create if does not exist.
        schema : string, default: None
            Name of SQL schema in database to write to (if database flavor
            supports this). If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default: None
            If not None, then rows will be written in batches of this size at a
            time.  If None, all rows will be written at once.
        dtype : single type or dict of column name to SQL type, default: None
            Optional specifying the datatype for columns. The SQL type should
            be a SQLAlchemy type. If all columns are of the same type, one
            single value can be used.

        """
        if dtype and not _is_dict_like(dtype):
            dtype = {col_name: dtype for col_name in catalog}

        if dtype is not None:
            for col, my_type in dtype.items():
                if not isinstance(to_instance(my_type), TypeEngine):
                    raise ValueError('The type of %s is not a SQLAlchemy '
                                     'type ' % col)

        table = SQLTable(name, self, catalog=catalog, if_exists=if_exists,
                         schema=schema, dtype=dtype)
        table.create()
        table.insert(chunksize)
        if (not name.isdigit() and not name.islower()):
            # check for potentially case sensitivity issues
            # Only check when name is not a number and name is not lower case
            engine = self.connectable.engine
            with self.connectable.connect() as conn:
                table_names = engine.table_names(
                    schema=schema or self.meta.schema,
                    connection=conn,
                )
            if name not in table_names:
                msg = (
                    "The provided table name '{0}' is not found exactly as "
                    "such in the database after writing the table, possibly "
                    "due to case sensitivity issues. Consider using lower "
                    "case table names."
                ).format(name)

    @property
    def tables(self):
        return self.meta.tables

    def table_exists(self, name, schema=None):
        return self.connectable.run_callable(
            self.connectable.dialect.has_table,
            name,
            schema or self.meta.schema,
        )

    def get_table(self, table_name, schema=None):
        schema = schema or self.meta.schema
        if schema:
            tbl = self.meta.tables.get('.'.join([schema, table_name]))
        else:
            tbl = self.meta.tables.get(table_name)

        # Avoid casting double-precision floats into decimals
        for column in tbl.columns:
            if isinstance(column.type, Numeric):
                column.type.asdecimal = False

        return tbl

    def drop_table(self, table_name, schema=None):
        schema = schema or self.meta.schema
        if self.table_exists(table_name, schema):
            self.meta.reflect(only=[table_name], schema=schema)
            self.get_table(table_name, schema).drop()
            self.meta.clear()

    def _create_sql_schema(self, catalog, table_name, keys=None, dtype=None):
        table = SQLTable(table_name, self, catalog=catalog, keys=keys,
                         dtype=dtype)
        return str(table.sql_schema())


def get_schema(catalog, name, keys=None, con=None, dtype=None):
    """
    Get the SQL db table schema for the given catalog.

    Parameters
    ----------
    catalog : afw table
    name : string
        name of SQL table
    keys : string or sequence, default: None
        columns to use a primary key
    con: an open SQL database connection object or a SQLAlchemy connectable
        Using SQLAlchemy makes it possible to use any DB supported by that
        library, default: None
    dtype : dict of column name to SQL type, default: None
        Optional specifying the datatype for columns. The SQL type should
        be a SQLAlchemy type.

    """

    sql_io = _sql_io_builder(con=con)
    return sql_io._create_sql_schema(catalog, name, keys=keys, dtype=dtype)

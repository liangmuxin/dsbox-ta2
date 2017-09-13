import pandas as pd
from dsbox.executer.executionhelper import NestedData

class FlattenTable(object):
    def joinedTable(self, df):
        df = df.copy()
        for index_col, value in enumerate(df.iloc[0]):
            if not isinstance(value, NestedData):
                continue
            nested_data_column = df.iloc[:,index_col]
            nested_table = nested_data_column[0].nested_data
            index_col_name = nested_data_column[0].index_column
            filename_col_name = nested_data_column[0].filename_column
            filenames = [x.filename for x in nested_data_column]
            df[filename_col_name] = filenames
            df = pd.merge(df, nested_table, on=index_col_name)
        return df

    def fit(self, df, label=None):
        nested = [isinstance(x, NestedData) for x in df.iloc[0] ]
        if not True in nested:
            return df
        return self.joinedTable(df)

    def transform(self, df, label=None):
        nested = [isinstance(x, NestedData) for x in df.iloc[0] ]
        if not True in nested:
            return df
        return self.joinedTable(df)

    def fit_transform(self, df, label=None):
        nested = [isinstance(x, NestedData) for x in df.iloc[0] ]
        if not True in nested:
            return df
        return self.joinedTable(df)

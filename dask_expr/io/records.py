from dask.utils import M


def to_records(df):
    return df.to_dask_dataframe().map_partitions(M.to_records)

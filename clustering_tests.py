#%%
import importlib
import load_data as ld
importlib.reload(ld)

#%%
data_raw_file = 'Data_Raw/Nov2018.csv'
cols_to_read = ['Timestamp (UTC_TIME)', 'IP.NSRCGEN:BIASDISCAQNV', 'IP.NSRCGEN:GASSASAQN', 'ITF.BCT25:CURRENT', 'IP.SOLCEN.ACQUISITION:CURRENT']
rows_to_read = list(range(500000, 1500000))

df = ld.read_data_from_csv(data_raw_file, cols_to_read, rows_to_read)
df = ld.convert_column_types(df)
df = ld.clean_data(df)

#%%

#%%

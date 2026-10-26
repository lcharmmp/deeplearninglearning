import pandas as pd
from pathlib import Path
DATA_FILE = Path(__file__).parent.parent / 'dsh_answer' /'stage1'/'ex2_pandas_csv_stats'/'house_prices.csv'
OUT_FILE = Path(__file__).parent/'outcsv'
print(f'DATA_FILE={DATA_FILE}')
df = pd.read_csv(DATA_FILE)
print(f'df.isna().sum()=\n{df.isna().sum()}')
print(f'df.isna.mean()=\n{df.isna().mean()}')
df_filled=df.fillna(df.mean())
df=df.fillna(0)
comparison=pd.DataFrame({
    'fill_0':df.mean(),
    'fill_mean':df_filled.mean()
})
print(comparison)
comparison.to_csv(OUT_FILE)


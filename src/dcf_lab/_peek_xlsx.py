import pandas as pd

p = 'dcf_suite_v020/combined_data_items_final.xlsx'
xl = pd.ExcelFile(p)
print('Sheets:', xl.sheet_names)
for s in xl.sheet_names[:5]:
    try:
        df = pd.read_excel(xl, s, nrows=10)
        print('\nSheet:', s, 'columns:', list(df.columns)[:20])
        print(df.head(3).to_dict(orient='records'))
    except Exception as e:
        print('Error reading', s, e)

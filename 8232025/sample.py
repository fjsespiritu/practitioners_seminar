#!/usr/bin/env python3
#!/usr/bin/python3
"""
    to run the file you might have to activate conda: conda activate base
    ample command:  python "C:\Users\Jojie\Documents\Practitioner's Seminar\8232025\sample.py" -f 20231206 -t ADBE COST CSCO MAR MCD -o "C:\Users\Jojie\Desktop\School\Masters\Sem 1\PracSem\review"
    choose only from these stocks: [ADBE, COST, CSCO, MAR, MCD]

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from typing import Dict, Union, Iterable, List, Optional
Datelike = Union[pd.Timestamp, str, int]
import argparse
from pathlib import Path

def date_fmt(date: Datelike, **kwargs) -> Dict:
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(str(date))

    fmt_dict = {
        'YYYY': date.strftime('%Y'),
        'MM': date.strftime('%m'),
        'DD': date.strftime('%d'),
        'YYYYMMDD': date.strftime('%Y%m%d'),
    }
    fmt_dict.update(kwargs)

    return fmt_dict

def read_data(file_path, date_range):
    out_data = []
    for d in date_range:
        out_data = out_data +  [pd.read_csv(file_path.format(**date_fmt(d)))] #<< USE date_fmt FUNCTION
    df_out = pd.concat(out_data).reset_index(drop = True)
    df_out['date'] = df_out['date'].apply(lambda x: pd.Timestamp(x))
    return df_out
    
def date_range(basis_path, start: Datelike, end: Datelike, format: str = '%Y%m%d', on_out_of_bounds: str = 'warn') -> pd.DatetimeIndex:
    basis_path = basis_path.format(YYYY='**', YYYYMMDD='**')
    files = glob.glob(basis_path)
    # str_dates = [re.search(r'\d{8}', x).group() for x in glob.glob(basis_path)]
    str_dates = [
    re.search(r'\d{8}', os.path.basename(f)).group()
    for f in files
    if re.search(r'\d{8}', os.path.basename(f))
]
    str_dates.sort()
    dates = pd.to_datetime(
        str_dates,
        format='%Y%m%d')
    
    start = pd.to_datetime(start, format=format)
    end = pd.to_datetime(end, format=format)
    if (start < pd.Timestamp('20000103')) or (end > dates.max()): 
        mesg = f"Can't create date ranges outside [{dates.min()}, {dates.max():%Y-%m-%d}]"
        if on_out_of_bounds == 'warn':
            print(f'{mesg}. Ignoring dates outside this range.')
            if start < dates.min():
                start = dates.min()
            if end > dates.max():
                end = dates.max()
        elif on_out_of_bounds == 'raise':
            raise ValueError(mesg)
    return dates[(start <= dates) & (dates <= end)]

def create_parser():
    """
    sample command: ./sample.py -f 20150104 -t ADBE -o /Users/rodbarit/Documents/Ateneo_seminar/2025/20250823/

    """
    p = argparse.ArgumentParser(description=f'Script to generate the daily prices files')
    p.add_argument('-f','--file_date', default=None, required=True,
                   help='Initial date of file to process in YYYYMMDD format.',
                  #type = str,
                  ) 
    p.add_argument( '-t','--ticker', nargs="+", default=None, required=True,
                   help=' one or more ticker names')   
    p.add_argument( '-o','--output_directory', default=None, required=True,
                   help='output directory path')    
    return p


if __name__ == '__main__' :
    parser = create_parser()
    kwargs = parser.parse_args()
    
    ticker = kwargs.ticker
    output_directory = kwargs.output_directory
    file_date = kwargs.file_date
    dte = pd.Timestamp(file_date)
    
    # ticker = ['ADBE']
    # dte = pd.Timestamp('20150102')
    # output_directory = "/Users/rodbarit/Documents/Ateneo_seminar/2025/20250823/"
    
    hist_start = dte + pd.DateOffset(days = -90)
    
    ##### this is the path where the daily files are located 
    path = "/Users/Jojie/Documents/Practitioner's Seminar/8232025/prices/prices/{YYYYMMDD}.prices.csv.gz" ### <------- must change to your path
    list_dates = date_range(path,hist_start,dte)
    
    df = read_data(path, list_dates)
    df.rename(columns={'stock splits':'stock_splits','industry_tag':'industry'},inplace=True)
    df = df[df.ticker.isin(ticker)]
    df = df.drop(columns=['country', 'brand_name'])    
    
    df['previous_close'] = df['close'].shift(1)
    df['simple_return']  = (df['close']+df['dividends']) / df['previous_close'] - 1
    df['log_return']    = np.log((df['close']+df['dividends'])/df['previous_close'])
    df['volatility_P21'] = df['log_return'].rolling(21).std()
    df['volatility_P60'] = df['log_return'].rolling(60).std()
    df['overnight_log_ret'] = np.log((df['open']+df['dividends'])/df['previous_close'])
    df['today_log_ret'] = np.log((df['close'])/df['open'])
    df['overnight_volatility_P21'] = df['overnight_log_ret'].rolling(21).std()
    df['today_volatility_P21'] = df['today_log_ret'].rolling(21).std()
    
    df = df[df['date']==dte].reset_index(drop=True)

    Path(f"{output_directory}/").mkdir(parents=True, exist_ok=True) ### creates the directory
    df.to_csv(f"{output_directory}/{dte.strftime('%Y%m%d')}.prices.csv.gz",
                                      index = False,
                                      header = True,
                                      compression = 'gzip')  

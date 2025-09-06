#!/tools/anaconda/redhat/7/2024.10/bin/python

import sys
sys.path.insert(0, "/q/phitopolis/datathon_research_interns/datathon_scripts")

import numpy as np
import pandas as pd
import argparse

import os
from typing import Tuple, List, Optional
import sys

from typing import Dict, Union, Iterable, List, Optional
Datelike = Union[pd.Timestamp, str, int]

from pathlib import Path
sys.path.append(os.getcwd())

import pyalpha
from pyalpha import pipeline_tools as pipetools
from pyalpha import transforms
from pyalpha import utils
from typing import Union, Dict
from datetime import datetime

def create_parser():
    p = argparse.ArgumentParser(description=f'Script to generate the daily logret files')
    p.add_argument('-f','--file_date', default=None, required=True,
                   help='Initial date of file to process in YYYYMMDD format.',
                  #type = str,
                  )
    p.add_argument( '-s','--sid', default=None, required=True, nargs='+',
                   help='cwiq_code')
    p.add_argument( '-o','--output_directory', default=None, required=True,
                   help='output directory path')
    p.add_argument('--debug', action='store_true', help="Set logging level to DEBUG.")

    return p

# Sample execution: python technicals.py -f 20160104 -s 10038243 10029434 10033653 10009241 10036975 -o /q/home/ph.frances..espiritu/ateneo_practitioner/technicals/

if __name__ == '__main__' :
    parser = create_parser()
    kwargs = parser.parse_args()

    output_directory = kwargs.output_directory
    dte    =  kwargs.file_date
    sids = [int(sid) for sid in kwargs.sid]

    PRICES_PATH         = pyalpha.PRICES_PATH

    dates_obj = utils.Dates()
    start_date = dates_obj.trading_day_offset(dte, -15)
    end_date = dates_obj.trading_day_offset(dte, 0)
    dates = dates_obj.date_range(start_date,end_date)
    dte = pd.Timestamp(dte)

    dte= pd.to_datetime(dte, format="%Y%m%d")
    dates = pd.to_datetime(dates)

    df_temp = utils.read_data(PRICES_PATH, dates)
    df_temp = df_temp[['date','cwiq_code','log_return','closing_price','splits','dividends']].copy()
    df_temp = df_temp.sort_values(['date']).reset_index(drop=True)
    df_temp = df_temp.rename(columns={'closing_price':'close_price'})

    df_temp['date'] = pd.to_datetime(df_temp['date'])

    results = []
    dte = pd.to_datetime(20160104, format="%Y%m%d")

    for s in sids:
        s_int = int(s)
        sid_df = df_temp[df_temp['cwiq_code'] == s_int].copy()

        #adjust prices
        sid_df['cumprod_splits'] = sid_df['splits'].cumprod()
        sid_df['adj_factor'] = sid_df['cumprod_splits']/sid_df['cumprod_splits'].iloc[-1]
        sid_df['adj_price'] = sid_df['close_price']*sid_df['adj_factor']
        sid_df['adj_div'] = sid_df['dividends']*sid_df['adj_factor']

        sid_df['logret'] = np.log((sid_df['adj_price']+sid_df['adj_div'])/sid_df['adj_price'].shift(1))
        sid_df['adj_price_div'] = sid_df['close_price']/np.exp(sid_df['logret'])

        # created an adjusted price based on log_ret - adjust prices to account for sudden drop in dividends

        #calculation of RSI
        sid_df['Up'] = np.where(sid_df['adj_price']-sid_df['adj_price'].shift(1) > 0, sid_df['adj_price']-sid_df['adj_price'].shift(), 0)
        sid_df['Down'] = np.where(sid_df['adj_price']-sid_df['adj_price'].shift(1) < 0, -(sid_df['adj_price']-sid_df['adj_price'].shift()), 0)

        sid_df['RS'] = (sid_df['Up'].rolling(14).mean())/(sid_df['Down'].rolling(14).mean())
        sid_df['RSI_14'] = 100 - (100/(1+sid_df['RS']))

        sid_df['MA10'] = sid_df['adj_price'].rolling(10).mean()

        #Select relevant columns
        cols = ['date', 'cwiq_code', 'MA10', 'RSI_14']
        final_df = sid_df[cols]

        final_df = final_df[final_df['date'] == dte]

        results.append(final_df)

    #Combine all stocks in one df
    new_df = pd.concat(results, ignore_index = True)
    
    Path(f"{output_directory}/{dte.strftime('%Y')}").mkdir(parents=True, exist_ok=True) ### creates the directory
    new_df.to_csv(f"{output_directory}/{dte.strftime('%Y')}/{dte.strftime('%Y%m%d')}.technicals.csv",
    index = False,
    header = True,
    )

    print("Writing files...")
    print("Done.")

                            

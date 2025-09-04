#!/tools/anaconda/redhat/7/2023.07/bin/python

import sys

sys.path.insert(0,
                "/q/phitopolis/datathon_research_interns/datathon_scripts")  # inserts the scripts to the PATH environment - so we can import the scripts in our notebook

import numpy as np
import pandas as pd
import argparse

import lightgbm as lgb
import matplotlib.pyplot as plt
import statsmodels.api as sm

import os
from typing import Tuple, List, Optional
import sys
import empyrical as em

from typing import Dict, Union, Iterable, List, Optional

Datelike = Union[pd.Timestamp, str, int]  # can be either a pandas timestamp, a string, or an integer

from pathlib import Path

sys.path.append(os.getcwd())

import pyalpha
from pyalpha import pipeline_tools as pipetools
from pyalpha import transforms
from pyalpha import utils
from typing import Union, Dict
from datetime import datetime

import logging

def setup_logging(debug_mode: bool):
    log_level = logging.DEBUG if debug_mode else logging.INFO

    # Configure basic logging to a file
    # The level is now determined by the command-line flag
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='log.log',
        filemode='a'  # 'a' for append, 'w' for overwrite
    )

    # Create a logger instance
    logger = logging.getLogger(__name__)

    # Configure a console handler to also show logs in the terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def create_parser():
    p = argparse.ArgumentParser(description=f'Script to generate the daily logret files')
    p.add_argument('-f', '--file_date', default=None, required=True,
                   help='Initial date of file to process in YYYYMMDD format.',
                   # type = str,
                   )
    p.add_argument('-s', '--sid', default=None, required=True, nargs='+',
                   help='cwiq_code')
    p.add_argument('-o', '--output_directory', default=None, required=True,
                   help='output directory path')
    p.add_argument('-w','--window', default= 25, required=True, type=int)
    p.add_argument('--debug', action='store_true', help="Set logging level to DEBUG.")

    return p

if __name__ == '__main__':
    parser = create_parser()
    kwargs = parser.parse_args()

    logger = setup_logging(kwargs.debug)
    output_directory = kwargs.output_directory
    dte = kwargs.file_date
    dte = pd.to_datetime(dte, format='%Y%m%d')
    
    sids = kwargs.sid
    window = kwargs.window

    dates_obj = utils.Dates()
    dates = dates_obj.date_range(start = dates_obj.trading_day_offset(dte,-window), end = dates_obj.trading_day_offset(dte,5))
    dates

    PRICES_PATH = pyalpha.PRICES_PATH
    price_df = utils.read_data(PRICES_PATH, dates, columns = ['date','cwiq_code','closing_price','splits','dividends'])
    price_df['date'] = pd.to_datetime(price_df['date'], format='%Y%m%d')

    MACROECONOMIC_PATH = pyalpha.MACROECONOMIC_PATH
    macros = utils.read_data(MACROECONOMIC_PATH, dates, columns = ['date', 'SPX'])
    macros['date'] = pd.to_datetime(macros['date'], format='%Y%m%d')
    macros['SPX_log_ret'] =  np.log(macros['SPX']/macros['SPX'].shift(1))

    results = []


    for s in sids:
        s_int = int(s)
        data = price_df[price_df['cwiq_code'] == s_int].copy()

        #Log-returns
        data['prev_adj_price'] = data['closing_price'].shift(1)/data['splits']
        data['log_return'] = np.log((data['closing_price']+data['dividends'])/(data['prev_adj_price']))

        #Log ret F1 to F5
        for k in range(1,6):
            data[f'log_ret_F{k}'] = data['log_return'].shift(-k)

        #Log ret SF5
        sf5_cols = ['log_ret_F1', 'log_ret_F2', 'log_ret_F3', 'log_ret_F4', 'log_ret_F5']
        data['log_ret_SF5'] = data[sf5_cols].sum(axis=1)

        # Select relevant columns
        df_cols = ['date', 'cwiq_code', 'log_return'] + sf5_cols + ['log_ret_SF5']
        daily_log_ret = data[df_cols]

        daily_log_ret_with_spx = daily_log_ret.merge(macros, on='date')

        daily_log_ret_with_spx = daily_log_ret_with_spx.sort_values('date')
        num = daily_log_ret_with_spx['log_return'].rolling(window).cov(daily_log_ret_with_spx['SPX_log_ret'])
        denom = daily_log_ret_with_spx['SPX_log_ret'].rolling(window).var()

        daily_log_ret_with_spx[f'beta_SPX_{window}2'] = num/denom

        daily_log_ret_with_spx= daily_log_ret_with_spx[daily_log_ret_with_spx['date'] == dte]

        results.append(daily_log_ret_with_spx)

    # Combine all stocks into one DataFrame
    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.drop(columns = ['SPX'])

    Path(f"{output_directory}/").mkdir(parents=True, exist_ok=True)
    final_df.to_csv(f"{output_directory}/{dte.strftime('%Y%m%d')}.beta.csv",index = False, header = True)  

    print(dte)
    print(output_directory)
    print(sids)

    logger.info("Writing files...")
    logger.info("Done.")

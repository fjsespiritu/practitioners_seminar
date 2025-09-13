#!/tools/anaconda/redhat/7/2023.07/bin/python

### alternatively, you can run the .py file using: /tools/anaconda/redhat/7/2023.07/bin/python ./template.py

import sys

sys.path.insert(0, "/q/phitopolis/datathon_research_interns/datathon_scripts")

import numpy as np
import pandas as pd
import argparse

import os
from pathlib import Path

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
                   help='cwiq_code', type=int)
    p.add_argument('-o', '--output_directory', default=None, required=True,
                   help='output directory path')
    p.add_argument('--debug', action='store_true', help="Set logging level to DEBUG.")

    return p


# Sample execution: ./long_exam.py -f 20160104 -s 10038243 10029434 10033653 10009241 10036975 -o /q/home/ph.frances.espiritu/ateneo_practitioner/long_exam/
### when debugging you can use pdb
# import pdb; pdb.set_trace()
# n - next
# c - continue
# df - you can print the df as you go along

if __name__ == '__main__':
    parser = create_parser()
    kwargs = parser.parse_args()

    logger = setup_logging(kwargs.debug)
    output_directory = kwargs.output_directory
    dte = kwargs.file_date
    sids = kwargs.sid

    PRICES_PATH = pyalpha.PRICES_PATH
    MACROECONOMIC_PATH = pyalpha.MACROECONOMIC_PATH

    print(dte, type(dte))
    print(sids, type(sids))
    print(output_directory, type(output_directory))

    # sids = [10038243, 10029434, 10033653, 10009241, 10036975]
    # date = "20160104"
    # output_directory = "/q/home/ph.frances.espiritu/ateneo_practitioner/long_exam/"

    ### dates
    dte = pd.to_datetime(dte)
    dates_obj = utils.Dates()
    dates = dates_obj.date_range(start = dates_obj.trading_day_offset(dte,-500), end = dates_obj.trading_day_offset(dte, 1))

    ### read prices file
    PRICES_PATH = pyalpha.PRICES_PATH
    price_df = utils.read_data(PRICES_PATH, dates, columns = ['date', 'cwiq_code', 'closing_price', 'splits', 'dividends','volume'])
    price_df['date'] = pd.to_datetime(price_df['date'], format = '%Y%m%d')

    #1. Compute for the turnover spike of a stock for the past 10 days.  
    #Turnover spike is equal to the volume for the current day divided by the average volume for the 
    #previous 10 days (exclude the current day).  After computing for 
    #the turnover spike, apply a rolling ZScore for the past 20 
    #days.  [25 pts]

    ### compute for the turnover spike
    price_df['shift_vol_f1'] = price_df.groupby('cwiq_code')['volume'].shift(1)
    price_df['ave_vol_10d'] = price_df.groupby('cwiq_code')['shift_vol_f1'].rolling(10).mean().droplevel('cwiq_code')
    price_df['turnover_spike10']  = price_df['volume']/price_df['ave_vol_10d']

    ### compute for the RollingZS20 of turnover spike
    price_df = transforms.rolling_zscore(price_df, cols = ['turnover_spike10'], window = 20)

    ### compute for logret
    results = []

    for s in sids:
        s_int = int(s)
        df = price_df[price_df['cwiq_code'] == s_int].copy()

        df['cumprod'] = df['splits'].cumprod()
        df['split_factor'] = df['cumprod']/df['cumprod'].iloc[-1]
        df['adj_price'] = df['closing_price']*df['split_factor']
        df['adj_div'] = df['dividends']*df['split_factor']

        df['logret'] = np.log((df['adj_price']+df['adj_div'])/df['adj_price'].shift(1))

       #2. Compute for the Amihud's illiquidity ratio for the past 252 days.  
       #For log_ret, don't use the existing one in the PRICES_PATH file. You need to compute 
        #for the log_ret by adjusting for splits and dividends.  Multiply the ratio to 
        #1,000,000,000 for readability because the figure will be a very small number.  
        #Add a ZSbydate transform after.  [25 pts]

         ### compute for amihud ratio
        df['adjusted_vol'] = df['volume'] * df['closing_price']

        df['daily_ratio'] = np.abs(df['logret'])/df['adjusted_vol']
        df['amihud_illiquidity252'] = 1000000000 * (df['daily_ratio'].rolling(252).sum())/252

        results.append(df)

    price_df_2 = pd.concat(results, ignore_index = True)

    ### compute for ZSbydate of amihud ratio
    price_df_2 = transforms.zscore_bydate(price_df_2, cols = ['amihud_illiquidity252'])

    ### read macro file
    macros = utils.read_data(MACROECONOMIC_PATH,dates, columns = ['date', 'SPX', 'T10YR'])

    ### sort values by date and forward fill the 10 year
    macros = macros.sort_values(by = ['date'], ascending = True)
    macros['T10YR'] = macros['T10YR'].ffill()

    #1. Estimate the beta of a stock with respect to the S&P 500 by 
    #computing for the beta using CAPM (expected_log_ret = rf + beta * (SP log_ret - rf)) 
    #where SP is the return of S&P 500 and rf is the risk free rate 
    #(using 10 year).  Residual return is the actual return reduced by the 
    #expected_log_ret. You will use 252 days of 1-day S&P 500 returns, 
    #1-day log returns, and US 10 Year (T10YR).

    ### compute spx log ret
    macros['SPX_logret'] = np.log(macros['SPX']/macros['SPX'].shift(1))

    ### compute daily return of T10YR - assume 252 days when annualizing
    macros['rf'] = ((1 + macros['T10YR']/100) ** (1/252))-1

    ### compute rm-rf
    macros['rmrf_diff'] = macros['SPX_logret'] - macros['rf']

    results2 = []

    for s in sids:
        s_int = int(s)
        df = price_df_2[price_df_2['cwiq_code'] == s_int].copy()

        ### merge the two dataframes
        merged_df = df.merge(macros, on = "date", how = "right")

        ### compute for the beta
        merged_df['num'] = merged_df['logret'].rolling(252).cov(merged_df['rmrf_diff'])
        merged_df['denom'] = merged_df['rmrf_diff'].rolling(252).var()
        merged_df['beta'] = merged_df['num']/merged_df['denom']

        merged_df['expected_log_ret'] = merged_df['rf'] + (merged_df['beta']*merged_df['rmrf_diff'])

        ### compute for the residual return
        merged_df['res_ret'] = merged_df['logret'] - merged_df['expected_log_ret']

        ### compute for the volatility of the residual returns
        merged_df['vol_resid20'] = merged_df['res_ret'].rolling(20).std()

        ### compute for the DF20 of residual volatility
        merged_df['vol_resid_DF20'] = merged_df['vol_resid20']-merged_df['vol_resid20'].shift(20)
        results2.append(merged_df)

    prices_with_macros = pd.concat(results2, ignore_index=True)

    ### compute for the ZSbydate of DF20 of residual volatility 
    prices_with_macros = transforms.zscore_bydate(prices_with_macros, cols = ['vol_resid_DF20'])

    ### filter to desired columns
    cols_to_keep = ['date', 'cwiq_code', 'turnover_spike10', 'turnover_spike10_RZS', 'amihud_illiquidity252', 'amihud_illiquidity252_ZS', 'beta', 'vol_resid20', 'vol_resid_DF20', 'vol_resid_DF20_ZS']
    final_df = prices_with_macros[cols_to_keep]
    final_df = final_df[final_df['date'] == dte]

    Path(f"{output_directory}/{dte.strftime('%Y')}").mkdir(parents=True, exist_ok=True) ### creates the directory
    final_df.to_csv(f"{output_directory}/{dte.strftime('%Y')}/{dte.strftime('%Y%m%d')}.long_exam.csv",index = False,
        header = True,
        )

    print(dte)
    print(output_directory)
    print(sids)
    logger.info("Done.")
import ccxt
import datetime
import pandas as pd
import os
import time
import sys
from time import sleep
from copy import deepcopy
import numpy as np
import random
import csv


def get_top_long_short_ratio(exchange, futures_symbol):
    while True:
        try:
            status = exchange.fapidataGetToplongshortaccountratio(
                {"symbol": futures_symbol, "period": '5m', "limit": 8})
            return status
        except Exception as e:
            print(f"get status: {e}")
            continue


def get_top_long_short_position_ratio(exchange, futures_symbol):
    while True:
        try:
            status = exchange.fapiDataGetTopLongShortPositionRatio(
                {"symbol": futures_symbol, "period": '5m', "limit": 8})
            return status
        except Exception as e:
            print(f"get status: {e}")
            continue

def get_depth_info(exchange, futures_symbol):
    while True:
        try:
            depth = exchange.fapiPublicGetDepth({"symbol": futures_symbol, "limit": 500})
            return depth['bids'], depth['asks']
        except Exception as e:
            print(f"Error fetching depth from server: {e}")
            continue
    return

def get_bid_ask_delta(exchange, futures_symbol):
    while True:
        try:
            depth = exchange.fapiPublicGetDepth({"symbol": futures_symbol, "limit": 500})
            return sum([float(x[1]) for x in depth['bids']]) - sum([float(x[1]) for x in depth['asks']])
            # return depth['bids'], depth['asks']
        except Exception as e:
            print(f"Error fetching depth from server: {e}")
            continue
    return

def get_bid_ask_value(exchange, futures_symbol):
    while True:
        try:
            depth = exchange.fapiPublicGetDepth({"symbol": futures_symbol, "limit": 500})
            return sum([float(x[1]) for x in depth['bids']]),  sum([float(x[1]) for x in depth['asks']]), float(depth['E'])
            # return depth['bids'], depth['asks']
        except Exception as e:
            print(f"Error fetching depth from server: {e}")
            continue
    return

def get_recent_trades(exchange, futures_symbol):
    while True:
        try:
            trades = exchange.fapiPublicGetTrades({"symbol": futures_symbol, "limit": 1000})
            return trades
        except Exception as e:
            print(f"Error fetching trades from server: {e}")
            continue
    return

def Get_Taker_longshortratio(exchange, futures_symbol):
    while True:
        try:
            status = exchange.fapidataGetTakerlongshortratio(
                {"symbol": futures_symbol, "period": '5m', "limit": 5})
            return status
        except Exception as e:
            print(f"get status: {e}")
            continue

def Get_openinterest_5min(exchange, futures_symbol):
    while True:
        try:
            status = exchange.fapidataGetOpeninteresthist({"symbol": futures_symbol, "period": '5m', "limit": 4})
            return status
        except Exception as e:
            print(f"get status: {e}")
            continue

def get_open_interest(exchange, futures_symbol):
    while True:
        try:
            status = exchange.fetchOpenInterest(futures_symbol)
            return float(status['openInterestAmount']), float(status['timestamp'])
        except Exception as e:
            print(f"get status: {e}")
            continue

def convert_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')

def is_first_second_of_minute(kline_server_time, minutes):
    # Convert server time from milliseconds to seconds
    server_time_seconds = kline_server_time / 1000.0 / (minutes)
    # Convert to a datetime object
    # dt = datetime.datetime.fromtimestamp(server_time_seconds)
    # Check if the minute is zero
    return int(server_time_seconds)

def sync_server_time(exchange):
    while True:
        try:
            server_time = exchange.fapiPublicGetTime()
            server_time = int(server_time['serverTime'])
            local_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
            return server_time - local_time, server_time
        except Exception as e:
            print(f"Error fetching time from server: {e}")
            continue
    return


def dataCollect():
    exchange = ccxt.binance({
        'apiKey': '',
        'secret': '',
        'enableRateLimit': True  # Recommended by ccxt when making frequent requests
    })

    # Define the trading pair and contract type
    futures_symbol = "ETHUSDT"
    contract_type = 'PERPETUAL'

    time_interval = 5 # 5 seconds
    filename = 'ETH_trade_data.csv'

    with open(filename, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(['datetime', 'price','taker_buy', 'taker_sell', 'bid_volume', 'ask_volume', 'open_interest'])
    First_second_of_interval = None
    offset, server_time = sync_server_time(exchange)
    LastTime = is_first_second_of_minute(server_time, time_interval)
    while True:
        trades = get_recent_trades(exchange, futures_symbol)
        latest_time = float(trades[-1]['time'])
        NewTime = is_first_second_of_minute(latest_time, time_interval)
        if NewTime != LastTime:
            First_second_of_interval = True
        else:
            First_second_of_interval = False
        #LastTime = NewTime
        if First_second_of_interval:

            #Taker long short value
            buy_volume = 0
            sell_volume = 0
            #loop trades reversely
            for trade in trades[::-1]:
                trade_time = float(trade['time'])
                if is_first_second_of_minute(trade_time, time_interval) == LastTime:
                    if trade['isBuyerMaker']:
                        sell_volume += float(trade['qty'])
                    else:
                        buy_volume += float(trade['qty'])
            price = float(trades[-1]['price'])
            
            sell_volume = round(sell_volume, 3)
            buy_volume = round(buy_volume, 3)
            # print(f"Time: {convert_timestamp(latest_time)}")

            #depth info
            bids, asks, depthtime = get_bid_ask_value(exchange, futures_symbol)
            bids = round(bids, 3)
            asks = round(asks, 3)
            # print(f"Depth Time: {convert_timestamp(depthtime)}")
            open_interest, OItime = get_open_interest(exchange, futures_symbol)
            # print(f"Open Interest Time: {convert_timestamp(OItime)}")
            with open(filename, mode='a') as file:
                writer = csv.writer(file)
                writer.writerow([convert_timestamp(latest_time),price, buy_volume, sell_volume, bids, asks, open_interest])

        LastTime = NewTime
        sleep(1)

if __name__ == "__main__":
    dataCollect()

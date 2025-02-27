import ccxt
import datetime
import pandas as pd
import os
import time
from time import sleep
from copy import deepcopy

import numpy as np
import random
import csv
import math
import ta
from Feature import *
from utils import *
import LorentzianModel
import GridLine as GL

from telegram.error import TelegramError
import asyncio
import signal
from telegram import Bot, Update
from telegram.ext import Updater
from DataCatch import dataCatch
import sys
from receieve_and_send import process_updates
from TakerEnergy import TakerEnergy

bot_token = ''
PS_chat_id = ''
GP_chat_id = ''
bot = Bot(token=bot_token)
##### Telegram command
Tele_com_auto_open_long_position = "open_auto_long"
Tele_com_auto_open_short_position = "open_auto_short"
Tele_com_auto_close_long_position = "close_auto_long"
Tele_com_auto_close_short_position = "close_auto_short"
Tele_com_close_position = "close"
Tele_com_get_balance = "balance"
Tele_com_make_all_none = "none"
Tele_com_open_volume_filter = "open_volume_filter"
Tele_com_close_volume_filter = "close_volume_filter"


async def backtest_strategy(exchange, leverage,
                            futures_symbol,
                            contract_type,
                            threshold_percentage, TP_SL_ratio,
                            loss_per_trade,
                            command_queue,
                            timeframe_merge,
                            use_Volume_filter,
                            shortlen,
                            longlen,
                            max_position_quantity_grid
                            ):
    timeframe_merge = int(timeframe_merge[:-1])
    starttime = time.time()
    timeframe = '1m'
    
    await send_text_message(bot=bot, msg=f"### Your Money is on the way ###",
                                chat_id=PS_chat_id)
    CostPrice = 2.41
    grid221 = None
    freeamount = 0
    USDTamount = 0
    
    Tele_adjust_loss_per_trade = loss_per_trade
    order_book = []
    long_order_confirm = []
    long_order_confirm2 = False
    short_order_confirm = []
    short_order_confirm2 = False
    BullBearFlip_confirm = []
    TP_price = None
    SL_price = None
    TP_order_list = None
    SL_order_list = None
    replaceSL = True
    isBullish = False
    isBearish = False
    closePosition = True
    initial_entry_price = None
    Signal_price = None

    maxProfit = 0
    Profitdown = 0.95
    sameKlineOpenPosition = True

    make_everthing_None = False
    Tele_auto_place_long_order = False
    Tele_auto_place_short_order = False
    Tele_auto_close_long_order = False
    Tele_auto_close_short_order = False
    Tele_close_all = False

    lastOpenLong = False
    lastOpenShort = False
    lastCloseLong = False
    lastCloseShort = False
    ReverseOrder = False    
    stopsend = True
    PositionOIstatus = None
   
    lr_train_buffer = await dataCatch(timeframe, 64000, futures_symbol)
    check_dimensions_initial(lr_train_buffer, 64000)
    print("Start to initialize the Lorentz Model")
    real_time_buffer, start_index = find_and_buffer(lr_train_buffer, timeframe_merge)
    resampled_df = resample_ohlc(lr_train_buffer, start_index - 1, timeframe_merge, 3999)
    buffer_ohlc = buffer_to_ohlc(real_time_buffer)
    final_df = pd.concat([resampled_df, buffer_ohlc], ignore_index=True)

    use_adx_filter = False
    LRmodel = LorentzianModel.LorentzianTradingModel(final_df, use_adx_filter, use_Volume_filter)
    LRmodel.useRegimeFilter = True
    LRmodel.Volume_shortlen = shortlen
    LRmodel.Volume_longlen = longlen
    endtime = time.time()
    initTime = endtime - starttime
    # convert initTime to minutes
    initTime = initTime / 60
    print("Initialize the Lorentz Model successfully")

    compensate_buffer = await dataCatch(timeframe, math.ceil(initTime) + 1, futures_symbol)

    offset, server_time = await sync_server_time(exchange)
    LastTime = is_first_second_of_minute(server_time, timeframe_merge)
    # print("Last", LastTime)
    Kline_first_second_of_minute = False
    for ind, row in compensate_buffer.iterrows():
        if row['timestamp'] == real_time_buffer.iloc[-1]['timestamp']:
            real_time_buffer.iloc[-1] = row
            break

    # lr_train_buffer = lr_train_buffer.iloc[2000:]
    final_df = final_df.iloc[2000:]
    for i in range(ind + 1, len(compensate_buffer)):
        # LRmodel.y_train_buffer = lr_train_buffer
        # LRmodel.process_row()
        if len(real_time_buffer) < timeframe_merge:
            real_time_buffer = pd.concat([real_time_buffer, compensate_buffer.iloc[i].to_frame().T], ignore_index=True)
        else:
            buffer_ohlc = buffer_to_ohlc(real_time_buffer)
            final_df.iloc[-1] = buffer_ohlc.iloc[-1]
            LRmodel.y_train_buffer = final_df
            LRmodel.process_row()
            if i < len(compensate_buffer):
                real_time_buffer = compensate_buffer.iloc[i:]
            break
    endtime2 = time.time()
    compensaTime = endtime2 - endtime

    while True:

        try:
            # command is (str) and sender_id is (int)
            command, sender_id = await asyncio.wait_for(command_queue.get(), timeout=1)
            if command and sender_id:
                if sender_id == int(PS_chat_id):
                    print(f"Received command: {command}")

                    if command.isdigit():
                        Tele_adjust_loss_per_trade = int(command)

                        await send_text_message(bot=bot,
                                                msg=f"### adjust loss per trade to {Tele_adjust_loss_per_trade}$ ###",
                                                chat_id=PS_chat_id)
                    elif command.lower() == Tele_com_auto_open_long_position.lower():
                        Tele_auto_place_long_order = True

                        await send_text_message(bot=bot, msg="### start auto open long position only! ###",
                                                chat_id=PS_chat_id)

                    elif command.lower() == Tele_com_auto_open_short_position.lower():
                        Tele_auto_place_short_order = True

                        await send_text_message(bot=bot, msg="### start auto open short position only! ###",
                                                chat_id=PS_chat_id)
                    elif command.lower() == Tele_com_auto_close_long_position.lower():
                        Tele_auto_close_long_order = True

                        await send_text_message(bot=bot, msg="### start auto close long position ###",
                                                chat_id=PS_chat_id)
                    elif command.lower() == Tele_com_auto_close_short_position.lower():
                        Tele_auto_close_short_order = True

                        await send_text_message(bot=bot, msg="### start auto close short position ###",
                                                chat_id=PS_chat_id)

                    elif command.lower() == Tele_com_close_position.lower():
                        Tele_close_all = True

                    # elif command.lower() == Tele_com_get_balance.lower():
                    #     await send_text_message(bot=bot, msg=f"### current balance: {balance} ###",
                    #                             chat_id=GP_chat_id)
                    elif command.lower() == Tele_com_make_all_none.lower():
                        make_everthing_None = True
                        await send_text_message(bot=bot, msg=f"### make all variables to its initial states ###",
                                                chat_id=PS_chat_id)
                    elif command.lower() == Tele_com_open_volume_filter.lower():
                        LRmodel.useVolume_Osc = True
                        await send_text_message(bot=bot, msg=f"### open volume filter ###",
                                                chat_id=PS_chat_id)
                    elif command.lower() == Tele_com_close_volume_filter.lower():
                        LRmodel.useVolume_Osc = False
                        await send_text_message(bot=bot, msg=f"### close volume filter ###",
                                                chat_id=PS_chat_id)

            # Process the command and adjust the strategy accordingly
            command_queue.task_done()
        except asyncio.TimeoutError:
            pass

        ############# close all ############################
        if Tele_close_all:
            Tele_close_all = False
            if order_book:
                # await close_all_position_at_order_book(exchange, order_book, new_close, balance, futures_symbol, bot,
                #                                        PS_chat_id)
                # order_book = []
                # _, server_time = await sync_server_time(exchange)
                # await cancel_ALL_orders(exchange, futures_symbol, server_time)
                initial_entry_price = None
            else:
                await send_text_message(bot=bot, msg="### Order book is empty, nothing to close ###",
                                        chat_id=PS_chat_id)

        offset, server_time = await sync_server_time(exchange)
        # balance = await get_current_balance(exchange, server_time)
        # max_position_size = balance * (leverage - 10)
        NewTime = is_first_second_of_minute(server_time, timeframe_merge)
        if NewTime != LastTime:
            Kline_first_second_of_minute = True
        else:
            Kline_first_second_of_minute = False
        LastTime = NewTime
        # 做一个容错就是如果这次fetch data 还是上一分钟的就re-fetch
        while True:

            new_data_1h = await dataCatch(timeframe, 5, futures_symbol)
            row_15 = new_data_1h.iloc[-1]

            if Kline_first_second_of_minute:
                while True:
                    if row_15['timestamp'] != real_time_buffer.iloc[-1]['timestamp']:
                        break
                    else:
                        sleep(0.5)
                        new_data_1h = await dataCatch(timeframe, 5, futures_symbol)
                        row_15 = new_data_1h.iloc[-1]


            if Kline_first_second_of_minute:

                real_time_buffer.iloc[-1] = new_data_1h.iloc[-2]
                real_time_buffer = filter_max_volume(real_time_buffer)
                buffer_ohlc = buffer_to_ohlc(real_time_buffer)
                final_df.iloc[-1] = buffer_ohlc.iloc[-1]
                LRmodel.y_train_buffer = final_df
                LRmodel.process_row()
                LRmodel.bars_held_count()

                # if LRmodel.isBullish:
                #     isBullish = True
                #     isBearish = False
                # elif LRmodel.isBearish:
                #     isBearish = True
                #     isBullish = False

                # real_time_buffer = row_15
                # remove all
                real_time_buffer = real_time_buffer.iloc[0:0]
                real_time_buffer = pd.concat([real_time_buffer, row_15.to_frame().T], ignore_index=True)
                buffer_ohlc = buffer_to_ohlc(real_time_buffer)
                final_df = pd.concat([final_df, buffer_ohlc], ignore_index=True)
                final_df = final_df.iloc[1:]
                sameKlineOpenPosition = True

            else:
                # Replace the last row in local_data_15_window with row_15
                real_time_buffer = pd.concat([real_time_buffer, row_15.to_frame().T], ignore_index=True)
                real_time_buffer = filter_max_volume(real_time_buffer)
                buffer_ohlc = buffer_to_ohlc(real_time_buffer)
                final_df.iloc[-1] = buffer_ohlc.iloc[-1]

            break

        """
                    every time check if the following condition been triggered
                        1. TP or SL triggerd
                        2. trend condition triggerd 
                """
        # if TP_order_list:

        #     TP_status = await get_order_status(exchange, futures_symbol, TP_order_list)
        #     # if TP_status['status'] == "FILLED" or TP_status['status'] == "CANCELED":
        #     if TP_status['status'] == "FILLED":
        #         print("TP Triggered")
        #         await send_text_message(bot=bot, msg="TP Triggered",
        #                                 chat_id=PS_chat_id)
        #         _, server_time = await sync_server_time(exchange)
        #         await cancel_ALL_orders(exchange, futures_symbol, server_time)
        #         order_book = []
        #         maxProfit = 0
        #         # cancel_orders(exchange, futures_symbol, SL_order_list, server_time)
        #         # make_everthing_None = True
        #         TP_order_list = None
        #         SL_order_list = None

        # if SL_order_list:

        #     SL_status = await get_order_status(exchange, futures_symbol, SL_order_list)
        #     # if SL_status['status'] == "FILLED" or SL_status['status'] == "CANCELED":
        #     if SL_status['status'] == "FILLED":
        #         print("SL Triggered")
        #         await send_text_message(bot=bot, msg="SL Triggered",
        #                                 chat_id=PS_chat_id)

        #         _, server_time = await sync_server_time(exchange)
        #         await cancel_ALL_orders(exchange, futures_symbol, server_time)
        #         order_book = []
        #         maxProfit = 0

        #         # cancel_orders(exchange, futures_symbol, TP_order_list, server_time)
        #         TP_order_list = None
        #         SL_order_list = None

        # if make_everthing_None:
        #     _, server_time = await sync_server_time(exchange)
        #     await cancel_ALL_orders(exchange, futures_symbol, server_time)
        #     make_everthing_None = False
        #     Tele_auto_place_long_order = False
        #     Tele_auto_place_short_order = False
        #     Tele_auto_close_long_order = False
        #     Tele_auto_close_short_order = False
        #     Tele_close_all = False
        #     LRmodel.useVolume_Osc = False

        
        _, server_time = await sync_server_time(exchange)
        account_info = await get_spot_account_info(exchange, server_time)
        #find where the asset is XRP in the list
        target_symbol = futures_symbol.replace("USDC", "")

        for i in range(len(account_info['balances'])):
            if account_info['balances'][i]['asset'] == target_symbol:
                freeamount = float(account_info['balances'][i]['free'])
                break
        for i in range(len(account_info['balances'])):
            if account_info['balances'][i]['asset'] == "USDC":
                USDTamount = float(account_info['balances'][i]['free'])
                break

        
        date = convert_binance_server_time(server_time)
        new_close = final_df.iloc[-1]['close']
        new_volume = final_df.iloc[-1]['volume']
        OpenInterestLatest = await get_open_interest(exchange, futures_symbol)
        LRmodel.y_train_buffer = final_df
        res = LRmodel.process_row()
        # Filter_direaction = EMA_SMA_Filter(final_df, 200)
        # if res[0]:
        #     long_order_confirm.append(res[0])
        # else:
        #     long_order_confirm = []

        # if res[1]:
        #     short_order_confirm.append(res[1])
        # else:
        #     short_order_confirm = []

        # if len(long_order_confirm) == 1 or len(short_order_confirm) == 1:
        #     Signal_price = new_close

        with open('result15.txt', 'a') as f:
            f.write(str([date, res, list(LRmodel.Signal_array.values())[-1]]) + ' isBullish:' + str(
                LRmodel.isBullish) + ' isBearish:' + str(LRmodel.isBearish) + ' yhat1:' + str(
                list(LRmodel.yhat1_array.values())[-1]) + ' Barheld:' + str(LRmodel.bars_held) + '\n')

        yhat1 = list(LRmodel.yhat1_array.values())[-1]

        # Stop Grid
        if grid221 is not None:
            if signalType == "Long" and new_close < yhat1:
                grid221 = None
                await send_text_message(bot=bot, msg="### Spot Grid Stop! ###",
                                        chat_id=GP_chat_id)
            if signalType == "Short" and new_close > yhat1: 
                grid221 = None
                await send_text_message(bot=bot, msg="### Spot Grid Stop! ###",
                                        chat_id=GP_chat_id)


        if grid221 is not None:  
            trading_signal = grid221.get_trading_signal(new_close)
            if trading_signal == 'sell':
                if new_close > CostPrice:
                    order_type = "MARKET"

                    total_usd_value = new_close * freeamount + USDTamount
                    five_percent_value = total_usd_value * 0.005
                    num_coins = int(five_percent_value // new_close)
                    num_coins = max(num_coins, 3)

                    position_quantity = num_coins

                    if freeamount > position_quantity:
                        _, server_time = await sync_server_time(exchange)
                        await post_spot_order(exchange, futures_symbol, 'SELL', order_type, position_quantity, server_time, PS_chat_id, bot)
                        _, server_time = await sync_server_time(exchange)
                        account_info = await get_spot_account_info(exchange, server_time)
                        #find where the asset is XRP in the list
                        target_symbol = futures_symbol.replace("USDC", "")

                        for i in range(len(account_info['balances'])):
                            if account_info['balances'][i]['asset'] == target_symbol:
                                freeamount = float(account_info['balances'][i]['free'])
                                break
                        for i in range(len(account_info['balances'])):
                            if account_info['balances'][i]['asset'] == "USDC":
                                USDTamount = float(account_info['balances'][i]['free'])
                                break
                        balance = new_close * freeamount + USDTamount
                        await send_text_message(bot=bot, msg=f"### Sell complete, {target_symbol} Balance: {freeamount}, USDC Balance: {USDTamount}, Total Value: {new_close*freeamount + USDTamount} ###",
                                            chat_id=PS_chat_id)

            elif trading_signal == 'buy':
                #post buy order and update cost price
                order_type = "MARKET"

                total_usd_value = new_close * freeamount + USDTamount
                five_percent_value = total_usd_value * 0.005
                num_coins = int(five_percent_value // new_close)
                num_coins = max(num_coins, 3)

                position_quantity = num_coins
                if USDTamount >= (position_quantity * new_close * 1.05):

                    _, server_time = await sync_server_time(exchange)
                    order_id = await post_spot_order(exchange, futures_symbol, 'BUY',  order_type, position_quantity, server_time, PS_chat_id, bot)
                    order_status = await get_spot_order_status(exchange, str(order_id), futures_symbol)
                    _, server_time = await sync_server_time(exchange)
                    account_info = await get_spot_account_info(exchange, server_time)
                    #find where the asset is XRP in the list
                    target_symbol = futures_symbol.replace("USDC", "")

                    for i in range(len(account_info['balances'])):
                        if account_info['balances'][i]['asset'] == target_symbol:
                            freeamount = float(account_info['balances'][i]['free'])
                            break
                    for i in range(len(account_info['balances'])):
                        if account_info['balances'][i]['asset'] == "USDC":
                            USDTamount = float(account_info['balances'][i]['free'])
                            break
                    #update cost price
                    if freeamount == 0:
                        freeamount = position_quantity
                    CostPrice = (CostPrice * (freeamount-position_quantity) + order_status['cost'] * 1.003) / (freeamount)
                    balance = new_close*freeamount + USDTamount
                    #send message to telegram
                    await send_text_message(bot=bot, msg=f"### Buy complete, Cost Price updated to: {CostPrice}, {target_symbol} Balance: {freeamount}, USDC Balance: {USDTamount}, Total Value: {new_close*freeamount + USDTamount} ###",
                                            chat_id=PS_chat_id)
                    #save cost price into csv with time
                    with open('CostPrice.csv', mode='a') as file:
                        writer = csv.writer(file)
                        writer.writerow([date, CostPrice])



        #Signal!!!Get up and earn some money!
        if (res[0] or res[1]) and (grid221 is None):
            if res[0] != lastOpenLong or res[1] != lastOpenShort:
                await send_text_message(bot=bot, msg=f"### Spot Grid Start! ###",
                                        chat_id=GP_chat_id)

                #save signal time into csv
                offset, server_time = await sync_server_time(exchange)
                dataTime = datetime.datetime.fromtimestamp(server_time/1000).strftime('%Y-%m-%d %H:%M:%S')
                with open('BTCSignalTime.csv', mode='a') as file:
                    writer = csv.writer(file)
                    writer.writerow([dataTime])

                #initialize the grid
                initial_entry_price = new_close
                grid_size = round(initial_entry_price * 221.87 / 60000, 4)
                grid221 = GL.GridLine(initial_entry_price, grid_size, 1000)
                
                if res[0]:
                    signalType = "Long"
                if res[1]:
                    signalType = "Short"


        # await asyncio.sleep(0.1)
        lastOpenLong = res[0]
        lastOpenShort = res[1]



async def main():
    # Initialize CCXT for Binance
    exchange = ccxt.binance({
        'apiKey': '',
        'secret': '',
        'enableRateLimit': True  # Recommended by ccxt when making frequent requests
    })

    # Define the trading pair and contract type
    futures_symbol = "XRPUSDC"
    contract_type = 'PERPETUAL'
    leverage = 100
    threshold_percentage = 0.08
    TP_SL_ratio = 8.88
    loss_per_trade = 3
    shortlen = 15
    longlen = 45
    timeframe = '15m'
    use_Volume_filter = False
    max_position_quantity_grid = 0.03
    griz_size_percentage = 0.0
    update_queue = asyncio.Queue()
    command_queue = asyncio.Queue()
    updater = Updater(bot=bot, update_queue=update_queue)
    async with updater:
        updater_task = asyncio.create_task(updater.start_polling())
        processing_task = asyncio.create_task(process_updates(update_queue, bot, command_queue, PS_chat_id))
        trading_task = asyncio.create_task(
            backtest_strategy(exchange, leverage=leverage,
                              futures_symbol=futures_symbol,
                              contract_type=contract_type,
                              threshold_percentage=threshold_percentage, TP_SL_ratio=TP_SL_ratio,
                              loss_per_trade=loss_per_trade,
                              command_queue=command_queue,
                              timeframe_merge=timeframe,
                              use_Volume_filter=use_Volume_filter,
                              shortlen=shortlen,
                              longlen=longlen,
                              max_position_quantity_grid=max_position_quantity_grid
                              ))

        try:
            await asyncio.gather(updater_task, processing_task, trading_task)
        except asyncio.CancelledError:
            print("Tasks were cancelled.")
        finally:
            # Ensure this code runs after the tasks are cancelled
            print("Executing code after the tasks...")

    # asyncio.run(backtest_strategy(exchange, futures_symbol=futures_symbol, contract_type=contract_type, bot=bot, chat_id=chat_id, timeframe_merge=timeframe))


def handle_signal():
    loop.stop()


if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # for sig in (signal.SIGINT, signal.SIGTERM):
    #     loop.add_signal_handler(sig, loop.stop)
    # try:
    #     loop.run_until_complete(main())
    # finally:
    #     print("Shutting down...")
    loop = asyncio.get_event_loop()
    try:
        if sys.platform == 'win32':
            loop.add_signal_handler(signal.SIGINT, handle_signal)
        else:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, handle_signal)
        loop.run_until_complete(main())
    except NotImplementedError:
        if sys.platform == 'win32':
            for sig in (signal.SIGINT, signal.SIGBREAK):
                signal.signal(sig, handle_signal)
        loop.run_until_complete(main())
    finally:
        print("Shutting down...")
        loop.close()

import numpy as np
import pandas as pd
from Feature import ta_rsi, ta_wt, ta_cci, n_adx, normalize, filter_volatility, regime_filter, adx_filter, rational_quadratic, gaussian, volume_oscillator  # Assuming these functions are defined elsewhere

class LorentzianTradingModel:
    def __init__(self, df, useADXFilter, use_Volume_filter):
        self.neighborsCount = 8
        self.maxBarsBack = 2000
        self.featureCount = 5

        self.DirectionLabel = [1, -1, 0]  # 1: long, -1: short, 0: neutral

        self.RSIparamA = 18
        self.RSIparamB = 1
        self.WTparamA = 10
        self.WTparamB = 11
        self.CCIparamA = 20
        self.CCIparamB = 1
        self.ADXparamA = 20

        self.RSI2paramA = 9
        self.RSI2paramB = 1

        self.useVolatility = True
        self.useRegimeFilter = True
        self.useADXFilter = useADXFilter
        self.useVolume_Osc = use_Volume_filter
        self.regimeThreshold = -0.1
        self.ADXThreshold = 20

        self.bars_held = 0
        self.Volume_shortlen = 30
        self.Volume_longlen = 90
        self.volatility = True
        self.regime = True
        self.ADX = True

        # Kernel parameters
        self.h = 8
        self.r = 8
        self.x = 25
        self.lag = 2
        self.useKernelFilter = True
        self.useKernelSmoothing = False
        self.isDifferentSignalType = False

        # Data buffers
        # self.buffer_size = 2000
        # self.df = df.iloc[self.buffer_size:]
        # self.y_train_buffer = df
        self.y_train_buffer = df[0:1999]
        self.y_train_buffer2 = df[2000:]

        wt2000 = ta_wt(self.y_train_buffer, self.WTparamA, self.WTparamB)
        self.wt2000min = np.nanmin(wt2000)
        self.wt2000max = np.nanmax(wt2000)

        cci2000 = ta_cci(self.y_train_buffer['close'], self.y_train_buffer['close'], self.y_train_buffer['close'], self.CCIparamA, self.CCIparamB)
        self.cci2000min = np.nanmin(cci2000)
        self.cci2000max = np.nanmax(cci2000)

        self.y_train_array = {}
        self.RSI_array = {}
        self.WT_array = {}
        self.CCI_array = {}
        self.ADX_array = {}
        self.RSI2_array = {}
        self.distances = []
        self.predictions = []
        # self.ATR_array = []
        self.Signal_array = {}
        self.isBullish = False
        self.isBearish = False

        # self.is_held_four_bars = False
        # self.is_held_less_than_four_bars = False
        # self.isNewBuySignal = False
        # self.isNewSellSignal = False

        self.Regime_array = {}
        self.yhat1_array = {}
        self.yhat2_array = {}
        self.startLongTrade_array = {}
        self.startShortTrade_array = {}
        self.endLongTrade_array = {}
        self.endShortTrade_array = {}
        self.prediction_array = {}
        # self.d_first = []
        # self.d_last = []
        self.barIndex = 200
        for ind, row in self.y_train_buffer2.iterrows():
            self.y_train_buffer = pd.concat([self.y_train_buffer, row.to_frame().T], ignore_index=True)
            self.process_row()
            self.y_train_buffer = self.y_train_buffer.iloc[1:]





                

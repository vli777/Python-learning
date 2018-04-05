import math
import datetime
import numpy as np
import scipy.stats as stats
import talib as ta
from pytz import timezone
from sqlalchemy import or_
import pandas as pd

# Fortune 500 companies
def before_trading_start(context, data):
    sp_500 = get_fundamentals(
                query()
                #.filter(fundamentals.valuation.market_cap > 4e9)
                .filter(fundamentals.valuation.market_cap > 3e9)
                .filter(fundamentals.valuation.market_cap < 5e9)
                .filter(fundamentals.share_class_reference.is_primary_share == True)
                .filter(fundamentals.company_reference.country_id == "USA")
                .filter(or_(fundamentals.company_reference.primary_exchange_id == "NAS", fundamentals.company_reference.primary_exchange_id == "NYS"))
                .order_by(fundamentals.valuation.market_cap.desc())
                .limit(299))  # S&P 500 has 500 but I need 1 for the SPY  
    context.fundamental_df = sp_500
    update_universe(context.fundamental_df.columns.values)

def initialize(context):
    context.safe =   [ sid(23921),  # TLT 20+ Year T Bonds
                       sid(23870),  # IEF 7-10 Year T Notes
                       #sid(40779),  # HDGE
                       #sid(36385),  # SEF
                       #sid(32267),  # DOG
    ]
    context.secs =   [ sid(19662),  # XLY Consumer Discrectionary SPDR Fund   
                       sid(19656),  # XLF Financial SPDR Fund  
                       sid(19658),  # XLK Technology SPDR Fund  
                       sid(19655),  # XLE Energy SPDR Fund  
                       sid(19661),  # XLV Health Care SPRD Fund  
                       sid(19657),  # XLI Industrial SPDR Fund  
                       sid(19659),  # XLP Consumer Staples SPDR Fund   
                       sid(19654),  # XLB Materials SPDR Fund  
                       sid(19660)]  # XLU Utilities SPRD Fund
    context.spy_sid = sid(8554)  
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.25, price_impact=0.1))
    schedule_function(trade,date_rules.month_start(),time_rules.market_open())
    
    c = context
    c.stocks            = ''
    c.spy               = context.spy_sid
    c.history_close     = None
    c.history_max       = None
    c.history_min       = None
    c.lookback          = 62 
    c.smoothing         = 40     # 40 is standard
    c.holdtilpositive   = False
    c.holdtilnegative   = False
    c.buytomorrow       = False
    c.threshold         = 0.0035 # 5% is standard
    c.hilo_index        = []
    c.hilo_MAindex      = []
    c.slope_index       = []
    c.spyslope_index    = []
    c.current_hilo      = -10
    c.m                 = len(context.stocks)
    c.b_t               = np.ones(context.m) / context.m
    c.eps               = 0.75 # change epsilon here
    c.init              = False
    c.previous_datetime = None
    #for making the history available in before trade start:
    schedule_function(Algo_hilo, date_rules.every_day(), time_rules.market_close(minutes=5))
    # schedule_function(daily, date_rules.week_start(2), time_rules.market_open(minutes=30))
    schedule_function(info, date_rules.every_day(), time_rules.market_close())
    
    c.cash_low = c.portfolio.starting_cash    # for info()
    c.max_lvrg = 0
    c.max_shrt = 0
    c.risk_hi  = 0
    c.date_prv = ''
    c.date_end = str(get_environment('end').date())
    print '{} to {}  {}'.format(str(get_datetime().date()) , c.date_end, int(c.cash_low))
    
def trade(context, data):
    #exchange_time = pd.Timestamp(get_datetime()).tz_convert('US/Eastern')   
    num = 120    
    spy_z = (data[context.spy_sid].price - data[context.spy_sid].mavg(num)) \
        / data[context.spy_sid].stddev(num)  
    
    for bond in context.safe:    ## Risk on/off logic:
        if get_open_orders(bond): continue 
        if spy_z < -1.0:
            order_target_percent(bond, .25 / 2.)
            #log.info("Risk OFF: allocate %s" % (bond.symbol) + " at %s" % str(exchange_time))
        else:
            order_target(bond, 0)
            #log.info("Risk ON: zero weight %s" % (bond.symbol) + " at %s" % str(exchange_time))
      
    for stock in context.secs:
        if get_open_orders(stock): continue 
        sect_z = (data[stock].price - data[stock].mavg(num)) / data[stock].stddev(num) 
  
        if sect_z < spy_z and (1.0 > sect_z > -1.0):        ## sector trade logic   
            order_target_percent(stock, .13 / 2.)  
            #log.info("Allocate %s" % (stock.symbol) + " at %s" % str(exchange_time)) 
        else: 
            order_target(stock, 0)
            #log.info("Zero weight %s" % (stock.symbol) + " at %s" % str(exchange_time))
            
def Algo_hilo(context, data):
    c = context
    #cash     = c.portfolio.cash
    #leverage = c.account.leverage
    c.history_close = history(202, '1d', 'price')
    x1 = list(xrange(22))
    spyslope, spyintercept, psyr_value, spyp_value, spystd_err = stats.linregress(
        x1,c.history_close[c.spy][-22:])    # spy slope info
    c.spyslope_index.append(spyslope)
    z_spyslope  = stats.zscore(c.spyslope_index, axis=0, ddof=1)[-1]    # zscore of spy slope
    accspyslope = 0
    # accelleration of spy
    if len(c.spyslope_index) > 5:
        x2 = list(xrange(5))
        accspyslope, accspyintercept, accpsyr_value, accspyp_value, accspystd_err = stats.linregress(x2,c.spyslope_index[-5:])
    # long and short mavg spy
    #mavg_short    = np.mean(c.history_close[c.spy][-22:])
    mavg_long     = np.mean(c.history_close[c.spy][-200:])
    current_price = c.history_close[c.spy][-1]        
    
    # maxes and minimums
    c.history_max = c.history_close.idxmax(axis=0, skipna=True)
    c.history_min = c.history_close.idxmin(axis=0, skipna=True)
    minctr   = 0
    maxctr   = 0
    stockctr = len(c.history_close.columns)

    for stock in c.history_max.index:    # number of hi's and Lows
        datemax = c.history_max[stock]
        if str(type(datemax)) == "<class 'pandas.tslib.Timestamp'>" \
          and datemax.date() == get_datetime().date(): maxctr += 1
    for stock in c.history_min.index:
         datemin = c.history_min[stock] 
         if str(type(datemin)) == "<class 'pandas.tslib.Timestamp'>" \
           and datemin.date() == get_datetime().date(): minctr += 1          
    #hi = float(maxctr) / stockctr
    #lo = float(minctr) / stockctr
    c.current_hilo = float(maxctr - minctr) / stockctr
    c.hilo_index.append(c.current_hilo)
    z_ratio  = stats.zscore(c.hilo_index[-c.smoothing:], axis=0, ddof=1)[-1] # zscore current_hilo
    ma_ratio = np.mean(c.hilo_index[-c.smoothing:])
    c.hilo_MAindex.append(ma_ratio)
    if len(c.hilo_MAindex)> 7:    # calc slope of MA HiLo index
        x = list(xrange(7))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x, c.hilo_MAindex[-7:])
    else:
        slope   = 0
        #p_value = 0
        
    c.slope_index.append(slope)
    
    if len(c.slope_index) > 7:
        slope_ma = np.mean(c.slope_index[-7:])
    else:
        slope_ma = 0

    '''
    if mavg_long > current_price: 
        record(mavg_long_vs_price=-1) 
    else:
        record(mavg_long_vs_price=1)
    if c.holdtilpositive: 
        record(holdtilpositive= 1) 
    else: 
        record(holdtilpositive=0)   
    record(zspyslope = z_spyslope)
    record(zratio_hi_lo = z_ratio)
    record(slopespy = spyslope)    
    '''    
    if (spyslope > 0 and slope_ma > 0) and c.holdtilpositive:
        c.holdtilpositive = False
    elif spyslope < 0 and c.holdtilnegative:
        c.holdtilnegative = False
    
    if c.buytomorrow:
        order_target_percent(c.spy, 1 / 2.)
        c.buytomorrow = False
        return
         
    no_spys = c.portfolio.positions[c.spy].amount
    if  z_spyslope > 0:
        if 0 and no_spys == 0: 
            log.info('Buying: z score of the Slope of SPY over 22 days > 0: we are moving up')
        c.buytomorrow = True # Buy with delay of 1 day
    elif  z_spyslope < -.7:
        if 0 and no_spys == 0: 
            log.info('Buying: z score of the Slope of SPY over 22 days < -0.7: mean reverting')
        order_target_percent(c.spy, 1 / 2.)       
        c.holdtilpositive = True
    elif (z_spyslope < 0 and not c.holdtilpositive) or mavg_long > current_price or (z_spyslope > 1.0 or z_ratio > 1.5):
        '''
        if no_spys > 0 and mavg_long > current_price: 
            log.info('Selling: price below Long mavg 122 days')
        if no_spys > 0 and z_spyslope > 1.0: 
            log.info('Selling: z score of the Slope of SPY over 22 days > 1.0')
        if no_spys > 0 and z_ratio > 1.5: 
            log.info('Selling: z score of the Ratio HILO of SPY over 22 days > 1.5')
        if no_spys > 0 and (z_spyslope < 0 and not c.holdtilpositive): 
            log.info('Selling: z score of the Slope of SPY over 22 days < -0.0 and not holding till positive')
        '''
        order_target(c.spy, 0)

def handle_data(context, data):
    return    # possibly faster than pass?
    
def info(context, data):
    ''' Custom chart and/or log of profit_vs_risk returns and related information
    '''
    # # # # # # # # # #  Options  # # # # # # # # # #
    record_max_lvrg = 1          # maximum leverage encountered
    record_leverage = 0          # Leverage (context.account.leverage)
    record_q_return = 1          # Quantopian returns (percentage)
    record_pvr      = 1          # Profit vs Risk returns (percentage)
    record_pnl      = 0          # Profit-n-Loss
    record_shorting = 0          # Total value of any shorts
    record_risk     = 0          # Risked, maximum cash spent or shorts in excess of cash at any time
    record_risk_hi  = 1          # Highest risk overall
    record_cash     = 0          # Cash available
    record_cash_low = 1          # Any new lowest cash level
    logging         = 1          # Also log to the logging window conditionally (1) or not (0)
    log_method      = 'risk_hi'  # 'daily' or 'risk_hi'

    c = context                          # For brevity
    new_cash_low = 0                     # To trigger logging in cash_low case
    date = str(get_datetime().date())    # To trigger logging in daily case
    cash = c.portfolio.cash

    if int(cash) < c.cash_low:    # New cash low
        new_cash_low = 1
        c.cash_low   = int(cash)
        if record_cash_low:
            record(CashLow = int(c.cash_low))

    pvr_rtrn      = 0        # Profit vs Risk returns based on maximum spent
    q_rtrn        = 0        # Returns by Quantopian
    profit_loss   = 0        # Profit-n-loss
    shorts        = 0        # Shorts value
    start         = c.portfolio.starting_cash
    cash_dip      = int(max(0, start - cash))

    if record_cash:
        record(cash = int(c.portfolio.cash))  # Cash

    if record_leverage:
        record(Lvrg = c.account.leverage)     # Leverage

    if record_max_lvrg:
        if c.account.leverage > c.max_lvrg:
            c.max_lvrg = c.account.leverage
            record(MaxLvrg = c.max_lvrg)      # Maximum leverage

    if record_pnl:
        profit_loss = c.portfolio.pnl
        record(PnL = profit_loss)             # "Profit and Loss" in dollars

    for p in c.portfolio.positions:
        shrs = c.portfolio.positions[p].amount
        if shrs < 0:
            shorts += int(abs(shrs * data[p].price))

    if record_shorting:
        record(Shorts = shorts)               # Shorts value as a positve

    # Shorts in excess of cash to cover them, a positive value
    shorts_excess = int(shorts - cash) if shorts > cash else 0
    c.max_shrt    = int(max(c.max_shrt, shorts_excess))

    risk = int(max(cash_dip, shorts_excess, shorts))
    if record_risk:
        record(Risk = risk)                   # Amount in play, maximum of shorts or cash used

    new_risk_hi = 0
    if risk > c.risk_hi:
        c.risk_hi = risk
        new_risk_hi = 1

        if record_risk_hi:
            record(Risk_hi = c.risk_hi)       # Highest risk overall

    if record_pvr:      # Profit_vs_Risk returns based on max amount actually spent (risk high)
        if c.risk_hi != 0:     # Avoid zero-divide
            pvr_rtrn = 100 * (c.portfolio.portfolio_value - start) / c.risk_hi
            record(PvR = pvr_rtrn)        # Profit_vs_Risk returns

    if record_q_return:
        q_rtrn = 100 * (c.portfolio.portfolio_value - start) / start
        record(QRet = q_rtrn)                 # Quantopian returns to compare to pvr returns curve

    #from pytz import timezone
    if logging:
        if log_method == 'risk_hi' and new_risk_hi \
          or log_method == 'daily' and c.date_prv != date \
          or c.date_end == date \
          or new_cash_low:
            mxlv   = 'MaxLv '   + '%.1f' % c.max_lvrg   if record_max_lvrg else ''
            qret   = 'QRet '    + '%.1f' % q_rtrn       if record_q_return else ''
            pvr    = 'PvR_Ret ' + '%.1f' % pvr_rtrn     if record_pvr      else ''
            pnl    = 'PnL '     + '%.0f' % profit_loss  if record_pnl      else ''
            csh    = 'Cash '    + '%.0f' % cash         if record_cash     else ''
            csh_lw = 'CshLw '   + '%.0f' % c.cash_low   if record_cash_low else ''
            shrt   = 'Shrt '    + '%.0f' % shorts       if record_shorting else ''
            risk   = 'Risk '    + '%.0f' % risk         if record_risk     else ''
            rsk_hi = 'RskHi '   + '%.0f' % c.risk_hi    if record_risk_hi  else ''
            minute = get_datetime().astimezone(timezone('US/Eastern')).time().minute
            log.info('{} {} {} {} {} {} {} {} {} {}'.format(
                    minute, mxlv, qret, pvr, pnl, csh, csh_lw, shrt, risk, rsk_hi))

    if c.date_end == date:    # Log on last day, like cash 125199  portfolio 126890
        log.info('cash {}  portfolio {}'.format(
                int(cash), int(c.portfolio.portfolio_value)))

    c.date_prv = date

    
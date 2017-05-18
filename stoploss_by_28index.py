'''
二八小市值择时买卖

配置指定频率的调仓日，在调仓日每日指定时间，计算沪深300指数和中证500指数当前的20日涨
幅，如果2个指数的20日涨幅有一个为正，则进行选股调仓，之后如此循环往复。

止损策略：

    大盘止损：(可选)
        1. 每分钟取大盘前130日的最低价和最高价，如果最高大于最低的两倍则清仓，停止交易。
        2. 每分钟判断大盘是否呈现三只黑鸦止损，如果是则当天清仓并停止交易，第二天停止交
           易一天。

    个股止损：(可选)
        每分钟判断个股是否从持仓后的最高价回撤幅度，如果超过个股回撤阈值，则平掉该股持仓

    二八止损：(必需)
        每日指定时间，计算沪深300指数和中证500指数当前的20日涨幅，如果2个指数涨幅都为负，
        则清仓，重置调仓计数，待下次调仓条件满足再操作

版本：v2.0.7
日期：2016.11.15
作者：Morningstar
'''
import pandas as pd
import numpy as np
import tradestat
from datetime import datetime, timedelta
import scipy.optimize as sco  # 用于仓位控制最优化问题求解
import talib
from math import isnan, log

# 黑名单一览表，更新时间 2016.7.10 by 沙米
# 科恒股份、太空板业，一旦2016年继续亏损，直接面临暂停上市风险
def get_blacklist():
    blacklist = ["600656.XSHG", "300372.XSHE", "600403.XSHG", "600421.XSHG", "600733.XSHG", "300399.XSHE",
                 "600145.XSHG", "002679.XSHE", "000020.XSHE", "002330.XSHE", "300117.XSHE", "300135.XSHE",
                 "002566.XSHE", "002119.XSHE", "300208.XSHE", "002237.XSHE", "002608.XSHE", "000691.XSHE",
                 "002694.XSHE", "002715.XSHE", "002211.XSHE", "000788.XSHE", "300380.XSHE", "300028.XSHE",
                 "000668.XSHE", "300033.XSHE", "300126.XSHE", "300340.XSHE", "300344.XSHE", "002473.XSHE"]
    return blacklist

# 设置参数，初始化逻辑
def init(context):
    # 加载统计模块
    g.trade_stat = tradestat.trade_stat()

    # 1. 配置选股参数
    # 备选股票数目
    g.pick_stock_count = 100
    # 买入股票数目
    g.buy_stock_count = 5

    # 是否根据PE选股
    g.pick_by_pe = False
    # 如果根据PE选股，则配置最大和最小PE值
    if g.pick_by_pe:
        g.max_pe = 200
        g.min_pe = 0

    # 是否根据EPS选股
    g.pick_by_eps = True
    # 配置选股最小EPS值
    if g.pick_by_eps:
        g.min_eps = 0

    # 配置是否过滤创业板股票
    g.filter_gem = True
    # 配置是否过滤黑名单股票，回测建议关闭，模拟运行时开启
    g.filter_blacklist = False

    # 是否对股票评分
    g.is_rank_stock = True
    if g.is_rank_stock:
        # 参与评分的股票数目
        g.rank_stock_count = 20

    # 2. 配置止损参数
    # (1) 配置是否根据大盘历史价格止损
    # 大盘指数前130日内最高价超过最低价2倍，则清仓止损
    # 注：关闭此止损，收益增加，但回撤会增加
    g.is_market_stop_loss_by_price = True
    if g.is_market_stop_loss_by_price:
        # 配置价格止损判定指数，默认为上证指数，可修改为其他指数
        g.index_4_stop_loss_by_price = '000001.XSHG'

    # (2) 配置是否开启大盘三黑鸦止损
    # 个人认为针对大盘判断三黑鸦效果并不好，首先有效三只乌鸦难以判断，准确率实际来看也不好，
    # 其次，分析历史行情看一般大盘出现三只乌鸦的时候，已经严重滞后了，使用其他止损方式可能会更好
    g.is_market_stop_loss_by_3_black_crows = True
    if g.is_market_stop_loss_by_3_black_crows:
        # 配置三黑鸦判定指数，默认为上证指数，可修改为其他指数
        g.index_4_stop_loss_by_3_black_crows = '000001.XSHG'
        g.dst_drop_minute_count = 60

    # (3) 配置是否根据28指数值实时进行止损
    g.is_market_stop_loss_by_28_index = False
    # 二八指数
    # g.index2 = '000300.XSHG'  # 沪深300指数，表示二，大盘股
    # g.index8 = '000905.XSHG'  # 中证500指数，表示八，小盘股
    g.index2 = '000016.XSHG'  # 上证50指数
    g.index8 = '399333.XSHE'  # 中小板R指数
    # g.index8 = '399006.XSHE'  # 创业板指数
    # 判定调仓的二八指数20日增幅
    g.index_growth_rate = 0.01
    if g.is_market_stop_loss_by_28_index:
        # 配置当日28指数连续为跌的分钟计数达到指定值则止损
        g.dst_minute_count_28index_drop = 120

    # (4) 配置是否进行个股止损、止盈
    g.is_stock_stop_loss = False
    g.is_stock_stop_profit = False

    # (5) 重置当日止损参数，仅针对需要当日需要重置的参数
    reset_day_param()

    # 3. 配置调仓逻辑函数
    # 调仓日计数器，单位：日
    g.day_count = 0
    # 调仓频率，单位：日
    g.period = 3
    # 缓存股票持仓后的最高价
    g.last_high = {}
    # 每日收盘前10分钟，运行调仓函数
    scheduler.run_daily(do_rebalance, time_rule=market_close(minute=10))
    # 打印策略参数
    log_param()

# 重置当日参数，仅针对需要当日需要重置的参数
def reset_day_param():
    # 重置当日大盘价格止损状态
    if g.is_market_stop_loss_by_price:
        g.is_day_stop_loss_by_price = False
    # 重置三黑鸦状态分钟计时器
    if g.is_market_stop_loss_by_3_black_crows:
        g.cur_drop_minute_count = 0
    # 重置28指数止损分钟计时器
    if g.is_market_stop_loss_by_28_index:
        g.minute_count_28index_drop = 0
    # 清空当日个股250天内最大的3日涨幅的缓存
    if g.is_stock_stop_loss or g.is_stock_stop_profit:
        g.pct_change = {}
        #g.pct_change.clear()

# 开盘前判断市场行情
def before_trading(context):
    logger.info("---------------------------------------------")
    # 盘前判断三乌鸦状态，因为判断的数据为前4日
    g.is_last_day_3_black_crows = is_3_black_crows(g.index_4_stop_loss_by_3_black_crows)
    if g.is_last_day_3_black_crows:
        logger.info("==> 前4日已经构成三黑鸦形态")
    pass

# 择时控制，主要实现止损
def handle_bar(context, bar_dict):
    # 大盘价格止损
    if g.is_market_stop_loss_by_price:
        if market_stop_loss_by_price(context, g.index_4_stop_loss_by_price):
            return

    if g.is_market_stop_loss_by_3_black_crows:
        if market_stop_loss_by_3_black_crows(context, g.index_4_stop_loss_by_3_black_crows, g.dst_drop_minute_count):
            return

    if g.is_market_stop_loss_by_28_index:
        if market_stop_loss_by_28_index(context, g.dst_minute_count_28index_drop):
            return

    if g.is_stock_stop_loss:
        stock_stop_loss(context, bar_dict)

    if g.is_stock_stop_profit:
        stock_stop_profit(context, bar_dict)

# 2. 调仓，必须加择时，避免买入就赔的情况
def do_rebalance(context, bar_dict):
    logger.info("调仓日计数 [%d]" % (g.day_count))

    # 回看指数前20天的涨幅
    gr_index2 = get_growth_rate(g.index2)
    gr_index8 = get_growth_rate(g.index8)
    logger.info("当前%s指数的20日涨幅 [%.2f%%]" % (instruments(g.index2).symbol, gr_index2 * 100))
    logger.info("当前%s指数的20日涨幅 [%.2f%%]" % (instruments(g.index8).symbol, gr_index8 * 100))

    if gr_index2 <= g.index_growth_rate and gr_index8 <= g.index_growth_rate:
        clear_positions(context)
        g.day_count = 0
    else:  # if  gr_index2 > g.index_growth_rate or ret_index8 > g.index_growth_rate:
        if g.day_count % g.period == 0:
            logger.info("==> 满足条件进行调仓")
            buy_stocks = pick_stocks(context, bar_dict)
            logger.info("选股后可买股票: %s" % (buy_stocks))
            adjust_position(context, buy_stocks)
        g.day_count += 1

# 根据待买股票创建或调整仓位
# 对于因停牌等原因没有卖出的股票则继续持有
# 始终保持持仓数目为g.buy_stock_count
def adjust_position(context, buy_stocks):
    # 清仓不在买入清单中的股票
    for stock in context.portfolio.positions.keys():
        if stock not in buy_stocks:
            logger.info("stock [%s] in position is not buyable" % (stock))
            position = context.portfolio.positions[stock]
            close_position(position)
        else:
            logger.info("stock [%s] is already in position" % (stock))
    # 根据股票数量分仓
    # 此处只根据可用金额平均分配购买，不能保证每个仓位平均分配
    position_count = len(context.portfolio.positions)
    if g.buy_stock_count > position_count:
        value = context.portfolio.cash / (g.buy_stock_count - position_count)

        for stock in buy_stocks:
            if context.portfolio.positions[stock].quantity == 0:
                if open_position(stock, value):
                    if len(context.portfolio.positions) == g.buy_stock_count:
                        break

# 日交易结束后进行统计，并重置日参数
def after_trading(context):
    # 统计报告交易情况
    g.trade_stat.report(context)
    # 重置日止损参数
    reset_day_param()
    # 得到当前未完成订单
    orders = get_open_orders()
    for _order in orders:
        logger.info("canceled uncompleted order: %s" % (_order.order_id))
    pass

# 选取指定数目的小市值股票，再进行过滤，最终挑选指定可买数目的股票
def pick_stocks(context, bar_dict):
    q = query()

    if g.pick_by_pe:
        q = q.filter(
            fundamentals.eod_derivative_indicator.pe_ratio > g.min_pe,
            fundamentals.eod_derivative_indicator.pe_ratio < g.max_pe
        )

    if g.pick_by_eps:
        q = q.filter(
            fundamentals.financial_indicator.earnings_per_share > g.min_eps,
            # valuation.turnover_ratio > 3
        )

    q = q.order_by(fundamentals.eod_derivative_indicator.market_cap.asc()) \
        .limit(g.pick_stock_count)

    df = get_fundamentals(q)

    stock_list = list(df.columns.values)

    if g.filter_gem:
        stock_list = filter_gem_stock(context, stock_list)

    if g.filter_blacklist:
        stock_list = filter_blacklist_stock(context, stock_list)

    stock_list = filter_paused_stock(stock_list)
    stock_list = filter_st_stock(stock_list)
    stock_list = filter_limitup_stock(context, bar_dict, stock_list)
    stock_list = filter_limitdown_stock(context, bar_dict, stock_list)

    # 根据20日股票涨幅过滤效果不好，故注释
    # stock_list = filter_by_growth_rate(stock_list, 20)

    if g.is_rank_stock:
        # 若选出股票太多，则只取前g.rank_stock_count个进行排分
        if len(stock_list) > g.rank_stock_count:
            stock_list = stock_list[:g.rank_stock_count]

        # logger.debug("评分前备选股票: %s" %(stock_list))
        if len(stock_list) > 0:
            stock_list = rank_stocks(bar_dict, stock_list)
            # logger.debug("评分后备选股票: %s" %(stock_list))

    # 选取指定可买数目的股票
    if len(stock_list) > g.buy_stock_count:
        stock_list = stock_list[:g.buy_stock_count]
    return stock_list


# 5. 分级A基金轮动补仓
def fja_invest(context, bar_dict):
    '''
    cash = context.portfolio.cash
    min_stock= '150283.XSHE'  # 申万医药A
    min_discount= 0
    fja= [stk for stk in context.fja_list if bar_dict[stk].is_trading]
    for stock in fja:
        try:
            if bar_dict[stock].discount_rate < min_discount:
                min_stock= stock
                min_discount= bar_dict[stock].discount_rate
        except:
            pass
    if cash > 0:
        order_target_value(min_stock, cash)
        logger.info('买入%s,当前资产组合为%s' % (min_stock, str(context.portfolio.positions)))
    '''
    # 获得当日最小折价率基金代码及折价率
    discount_rate = pd.Series(data=np.nan, index=context.fja_list)
    for stk in discount_rate.index:
        try:
            discount_rate[stk] = bar_dict[stk].discount_rate
        except:
            pass
    discount_rate = discount_rate.dropna()
    if discount_rate.empty:
        return
    min_stock = discount_rate.argmin()
    min_discount = discount_rate[min_stock]
    # 第一次买入
    if context.cur_stock == '':
        shares = context.portfolio.cash / bar_dict[min_stock].close
        order_shares(min_stock, shares)
        logger.info("买入:" + min_stock + str(shares))
        context.cur_stock = min_stock
    else:
        # 如果当日最小折价率与当前持仓折价率相差超过1则轮仓
        cur_discount = bar_dict[context.cur_stock].discount_rate
        if context.cur_stock != min_stock and bar_dict[min_stock].is_trading \
                and bar_dict[context.cur_stock].is_trading and cur_discount - min_discount > 1:
            order_target_percent(context.cur_stock, 0)
            logger.info("卖出:" + context.cur_stock)
            shares = context.portfolio.cash / bar_dict[min_stock].close
            order_shares(min_stock, shares)
            logger.info("买入:" + min_stock + str(shares))
            context.cur_stock = min_stock


# 该仓位控制函数不错，值得拥有 (*****)
def update_weights(context, stocks):
    # 计算各支股票上一年的收益率
    # start_date = context.now + timedelta(days=-365)
    # end_date = context.now +timedelta(days =-1)
    # price= get_price(stocks, start_date, end_date, fields= ['close'])
    # rets= np.log(price/ price.shift(1))
    rets = get_price_change_rate(stocks, count=252)
    if rets is None:
        return 0

    # 根据股票池中股票过去一年的历史涨跌幅，算出平均收益和标准差，得到组合的收益、波动和夏普比率
    def statistics(weights):
        weights = np.array(weights)
        pret = np.sum(rets.mean() * weights) * 252
        pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
        return np.array([pret, pvol, pret / pvol])

    # 最优化目标函数：夏普比率最大
    def min_sharpe(weights):
        return -statistics(weights)[2]

    # 最优化目标函数：方差（波动率）
    def min_variance(weights):
        return statistics(weights)[1]

    # 约束条件：权重之和为1； 优化解的取值范围：0到1之间
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0, 1) for x in range(len(stocks)))
    # 求解各支股票的最优权重
    opts = sco.minimize(min_sharpe, len(stocks) * [1. / len(stocks)], bounds=bnds, \
                        method='SLSQP', constraints=cons)
    # 输出最优解中每支股票的权重，用于投资
    return opts['x'].round(3)


def log_param():
    logger.info("调仓日频率: %d日" % (g.period))
    logger.info("备选股票数目: %d" % (g.pick_stock_count))

    logger.info("是否根据PE选股: %s" % (g.pick_by_pe))
    if g.pick_by_pe:
        logger.info("选股最大PE: %s" % (g.max_pe))
        logger.info("选股最小PE: %s" % (g.min_pe))

    logger.info("是否根据EPS选股: %s" % (g.pick_by_eps))
    if g.pick_by_eps:
        logger.info("选股最小EPS: %s" % (g.min_eps))

    logger.info("是否过滤创业板股票: %s" % (g.filter_gem))
    logger.info("是否过滤黑名单股票: %s" % (g.filter_blacklist))
    if g.filter_blacklist:
        logger.info("当前股票黑名单：%s" % str(get_blacklist()))

    logger.info("是否对股票评分选股: %s" % (g.is_rank_stock))
    if g.is_rank_stock:
        logger.info("评分备选股票数目: %d" % (g.rank_stock_count))

    logger.info("买入股票数目: %d" % (g.buy_stock_count))

    logger.info("二八指数之二: %s - %s" % (g.index2, instruments(g.index2).symbol))
    logger.info("二八指数之八: %s - %s" % (g.index8, instruments(g.index8).symbol))
    logger.info("判定调仓的二八指数20日增幅: %.1f%%" % (g.index_growth_rate * 100))

    logger.info("是否开启大盘历史高低价格止损: %s" % (g.is_market_stop_loss_by_price))
    if g.is_market_stop_loss_by_price:
        logger.info("大盘价格止损判定指数: %s - %s" % (
            g.index_4_stop_loss_by_price, instruments(g.index_4_stop_loss_by_price).symbol))

    logger.info("大盘三黑鸦止损判定指数: %s - %s" % (
        g.index_4_stop_loss_by_3_black_crows, instruments(g.index_4_stop_loss_by_3_black_crows).symbol))
    logger.info("是否开启大盘三黑鸦止损: %s" % (g.is_market_stop_loss_by_3_black_crows))
    if g.is_market_stop_loss_by_3_black_crows:
        logger.info("三黑鸦止损开启需要当日大盘为跌的分钟计数达到: %d" % (g.dst_drop_minute_count))

    logger.info("是否根据28指数值实时进行止损: %s" % (g.is_market_stop_loss_by_28_index))
    if g.is_market_stop_loss_by_28_index:
        logger.info("根据28指数止损需要当日28指数连续为跌的分钟计数达到: %d" % (g.dst_minute_count_28index_drop))

    logger.info("是否开启个股止损: %s" % (g.is_stock_stop_loss))
    logger.info("是否开启个股止盈: %s" % (g.is_stock_stop_profit))


def market_stop_loss_by_price(context, index):
    # 大盘指数前160日内最高价超过最低价2.2倍，则清仓止损
    # 基于历史数据判定，因此若状态满足，则当天都不会变化
    # 增加此止损，回撤降低，收益降低

    if not g.is_day_stop_loss_by_price:
        h = history_bars(index, 160, '1d', fields=['close', 'high', 'low'])
        low_price = h['low'].min()
        high_price = h['high'].max()
        # if high_price > 2 * low_price:
        if high_price > 2.2 * low_price \
                and h['close'][-1] < h['close'][-4] \
                and h['close'][-1] > h['close'][-100]:
            # 当日第一次输出日志
            logger.info("==> 大盘止损，%s指数前160日内最高价超过最低价2.2倍, 最高价: %f, 最低价: %f" % (
                instruments(index).symbol, high_price, low_price))
            g.is_day_stop_loss_by_price = True

    if g.is_day_stop_loss_by_price:
        clear_positions(context)
        g.day_count = 0

    return g.is_day_stop_loss_by_price


def market_stop_loss_by_3_black_crows(context, index, n):
    # 前日三黑鸦，累计当日大盘指数涨幅<0的分钟计数
    # 如果分钟计数超过值n，则开始进行三黑鸦止损
    # 避免无效三黑鸦乱止损
    if g.is_last_day_3_black_crows:
        if get_growth_rate(index, 1) < 0:
            g.cur_drop_minute_count += 1

        if g.cur_drop_minute_count >= n:
            if g.cur_drop_minute_count == n:
                logger.info("==> 当日%s增幅 < 0 已超过%d分钟，执行三黑鸦止损" % (instruments(index).symbol, n))

            clear_positions(context)
            g.day_count = 0
            return True

    return False


def is_3_black_crows(stock):
    # talib.CDL3BLACKCROWS

    # 三只乌鸦说明来自百度百科
    # 1. 连续出现三根阴线，每天的收盘价均低于上一日的收盘
    # 2. 三根阴线前一天的市场趋势应该为上涨
    # 3. 三根阴线必须为长的黑色实体，且长度应该大致相等
    # 4. 收盘价接近每日的最低价位
    # 5. 每日的开盘价都在上根K线的实体部分之内；
    # 6. 第一根阴线的实体部分，最好低于上日的最高价位
    #
    # 算法
    # 有效三只乌鸦描述众说纷纭，这里放宽条件，只考虑1和2
    # 根据前4日数据判断
    # 3根阴线跌幅超过4.5%（此条件忽略）

    h = history_bars(stock, 4, '1d', ['close', 'open'])
    h_close = list(h['close'])
    h_open = list(h['open'])

    if len(h_close) < 4 or len(h_open) < 4:
        return False

    # 一阳三阴
    if h_close[-4] > h_open[-4] \
            and (h_close[-1] < h_open[-1] and h_close[-2] < h_open[-2] and h_close[-3] < h_open[-3]):
        # and (h_close[-1] < h_close[-2] and h_close[-2] < h_close[-3]) \
        # and h_close[-1] / h_close[-4] - 1 < -0.045:
        return True
    return False


'''
def is_3_black_crows(stock, data):
    # talib.CDL3BLACKCROWS
    his =  history_bars(stock, 2, '1d', ('close','open'), skip_paused=True, df=False)
    closeArray = list(his['close'])
    closeArray.append(data[stock].close)
    openArray = list(his['open'])
    openArray.append(get_current_data()[stock].day_open)

    if closeArray[0]<openArray[0] and closeArray[1]<openArray[1] and closeArray[2]<openArray[2]:
        if closeArray[-1]/closeArray[0]-1>-0.045:
            his2 =  history_bars(stock, 4, '1d', ('close','open'), skip_paused=True, df=False)
            closeArray1 = his2['close']
            if closeArray[0]/closeArray1[0]-1>0:
                return True
    return False
'''

def market_stop_loss_by_28_index(context, count):
    # 回看指数前20天的涨幅
    gr_index2 = get_growth_rate(g.index2)
    gr_index8 = get_growth_rate(g.index8)

    if gr_index2 <= g.index_growth_rate and gr_index8 <= g.index_growth_rate:
        if (g.minute_count_28index_drop == 0):
            logger.info("当前二八指数的20日涨幅同时低于[%.2f%%], %s指数: [%.2f%%], %s指数: [%.2f%%]" \
                        % (g.index_growth_rate * 100, instruments(g.index2).symbol, gr_index2 * 100,
                           instruments(g.index8).symbol, gr_index8 * 100))

            # logger.info("当前%s指数的20日涨幅 [%.2f%%]" %(instruments(g.index2).symbol, gr_index2*100))
            # logger.info("当前%s指数的20日涨幅 [%.2f%%]" %(instruments(g.index8).symbol, gr_index8*100))
        g.minute_count_28index_drop += 1
    else:
        # 不连续状态归零
        if g.minute_count_28index_drop < count:
            g.minute_count_28index_drop = 0

    if g.minute_count_28index_drop >= count:
        if g.minute_count_28index_drop == count:
            logger.info("==> 当日%s指数和%s指数的20日增幅低于[%.2f%%]已超过%d分钟，执行28指数止损" \
                        % (instruments(g.index2).symbol, instruments(g.index8).symbol,
                           g.index_growth_rate * 100, count))

        clear_positions(context)
        g.day_count = 0
        return True

    return False


# 个股止损，应用跟踪止损的思想
def stock_stop_loss(context, bar_dict):
    for stock in context.portfolio.positions.keys():
        cur_price = bar_dict[stock].close

        if g.last_high[stock] < cur_price:
            g.last_high[stock] = cur_price

        threshold = get_stop_loss_threshold(stock, g.period)
        # logger.debug("个股止损阈值, stock: %s, threshold: %f" %(stock, threshold))
        if cur_price < g.last_high[stock] * (1 - threshold):
            logger.info("==> 个股止损, stock: %s, cur_price: %f, last_high: %f, threshold: %f"
                        % (stock, cur_price, g.last_high[stock], threshold))

            position = context.portfolio.positions[stock]
            if close_position(position):
                g.day_count = 0


# 个股止盈
def stock_stop_profit(context, bar_dict):
    for stock in context.portfolio.positions.keys():
        position = context.portfolio.positions[stock]
        cur_price = bar_dict[stock].close
        threshold = get_stop_profit_threshold(stock, g.period)
        # logger.debug("个股止盈阈值, stock: %s, threshold: %f" %(stock, threshold))
        if cur_price > position.avg_price * (1 + threshold):
            logger.info("==> 个股止盈, stock: %s, cur_price: %f, avg_cost: %f, threshold: %f"
                        % (stock, cur_price, g.last_high[stock], threshold))

            if close_position(position):
                g.day_count = 0


# 获取个股前n天的m日增幅值序列
# 增加缓存避免当日多次获取数据
def get_pct_change(security, n, m):
    pct_change = None
    if security in g.pct_change.keys():
        pct_change = g.pct_change[security]
    else:
        h = history_bars(security, n, '1d', fields=('close'))
        pct_change = h['close'].pct_change(m)  # 3日的百分比变比（即3日涨跌幅）
        g.pct_change[security] = pct_change
    return pct_change


# 计算个股回撤止损阈值
# 即个股在持仓n天内能承受的最大跌幅
# 算法：(个股250天内最大的n日跌幅 + 个股250天内平均的n日跌幅)/2
# 返回正值
def get_stop_loss_threshold(security, n=3):
    pct_change = get_pct_change(security, 250, n)
    # logger.debug("pct of security [%s]: %s", pct)
    maxd = pct_change.min()
    # maxd = pct[pct<0].min()
    avgd = pct_change.mean()
    # avgd = pct[pct<0].mean()
    # maxd和avgd可能为正，表示这段时间内一直在增长，比如新股
    bstd = (maxd + avgd) / 2

    # 数据不足时，计算的bstd为nan
    if not isnan(bstd):
        if bstd != 0:
            return abs(bstd)
        else:
            # bstd = 0，则 maxd <= 0
            if maxd < 0:
                # 此时取最大跌幅
                return abs(maxd)

    return 0.099  # 默认配置回测止损阈值最大跌幅为-9.9%，阈值高貌似回撤降低


# 计算个股止盈阈值
# 算法：个股250天内最大的n日涨幅
# 返回正值
def get_stop_profit_threshold(security, n=3):
    pct_change = get_pct_change(security, 250, n)
    maxr = pct_change.max()

    # 数据不足时，计算的maxr为nan
    # 理论上maxr可能为负
    if (not isnan(maxr)) and maxr != 0:
        return abs(maxr)
    return 0.30  # 默认配置止盈阈值最大涨幅为30%


# 获取股票n日以来涨幅，根据当前价计算
# n 默认20日
def get_growth_rate(security, n=20):
    lc = get_close_price(security, n)
    # c = data[security].close
    c = get_close_price(security, 1, '1m')

    if not isnan(lc) and not isnan(c) and lc != 0:
        return (c - lc) / lc
    else:
        logger.error("数据非法, security: %s, %d日收盘价: %f, 当前价: %f" % (security, n, lc, c))
        return 0


# 获取前n个单位时间当时的收盘价
def get_close_price(security, n, unit='1d'):
    return history_bars(security, n, unit, 'close')[0]


# 开仓，买入指定价值的证券
# 报单成功并成交（包括全部成交或部分成交，此时成交量大于0），返回True
# 报单失败或者报单成功但被取消（此时成交量等于0），返回False
def open_position(security, value):
    order = order_target_value_(security, value)
    if order != None and order.filled_quantity > 0:
        # 报单成功并有成交则初始化最高价
        cur_price = get_close_price(security, 1, '1m')
        g.last_high[security] = cur_price
        return True
    return False


# 平仓，卖出指定持仓
# 平仓成功并全部成交，返回True
# 报单失败或者报单成功但被取消（此时成交量等于0），或者报单非全部成交，返回False
def close_position(position):
    security = position.order_book_id
    order = order_target_value_(security, 0)  # 可能会因停牌失败
    if order != None:
        if order.filled_quantity > 0:
            # 只要有成交，无论全部成交还是部分成交，则统计盈亏
            g.trade_stat.watch(security, order.filled_quantity, position.avg_price, position.market_value)

        if order.status == ORDER_STATUS.FILLED and order.filled_quantity == order.quantity:
            # 全部成交则删除相关证券的最高价缓存
            if security in g.last_high:
                g.last_high.pop(security)
            else:
                logger.warn("last high price of %s not found" % (security))
            return True

    return False


# 清空卖出所有持仓
def clear_positions(context):
    if context.portfolio.positions:
        logger.info("==> 清仓，卖出所有股票")
        for stock in context.portfolio.positions.keys():
            position = context.portfolio.positions[stock]
            close_position(position)

# 自定义下单
# 根据帮助文档，当前报单函数都是阻塞执行，报单函数（如order_target_value）返回即表示报单完成
# 报单成功返回报单（不代表一定会成交），否则返回None
def order_target_value_(security, value):
    if value == 0:
        logger.debug("Selling out %s" % (security))
    else:
        logger.debug("Order %s to value %f" % (security, value))

    # 如果股票停牌，创建报单会失败，order_target_value 返回None
    # 如果股票涨跌停，创建报单会成功，order_target_value 返回Order，但是报单会取消
    # 部成部撤的报单，聚宽状态是已撤，此时成交量>0，可通过成交量判断是否有成交
    return order_target_value(security, value)


# 过滤停牌股票
def filter_paused_stock(stock_list):
    return [stock for stock in stock_list if not is_suspended(stock)]


# 过滤ST及其他具有退市标签的股票
def filter_st_stock(stock_list):
    return [stock for stock in stock_list
            if not is_st_stock(stock)
            and 'ST' not in instruments(stock).special_type
            and '*' not in instruments(stock).special_type
            and '退' not in instruments(stock).symbol]


# 过滤涨停的股票
def filter_limitup_stock(context, bar_dict, stock_list):
    # 已存在于持仓的股票即使涨停也不过滤，避免此股票再次可买，但因被过滤而导致选择别的股票
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or bar_dict[stock].last < bar_dict[stock].limit_up]


# 过滤跌停的股票
def filter_limitdown_stock(context, bar_dict, stock_list):
    #
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or bar_dict[stock].last > bar_dict[stock].limit_down]


# 过滤黑名单股票
def filter_blacklist_stock(context, stock_list):
    blacklist = get_blacklist()
    return [stock for stock in stock_list if stock not in blacklist]


# 过滤创业版股票
def filter_gem_stock(context, stock_list):
    return [stock for stock in stock_list if instruments(stock).order_book_id[0:3] != '300']


# 过滤20日增长率为负的股票
def filter_by_growth_rate(stock_list, n):
    return [stock for stock in stock_list if get_growth_rate(stock, n) > 0]


# 股票评分排序
def rank_stocks(bar_dict, stock_list):
    dst_stocks = {}
    for stock in stock_list:
        h = history_bars(stock, 130, '1d', fields=['close', 'high', 'low'])
        low_price_130 = h['low'].min()
        high_price_130 = h['high'].max()

        # avg_15 = bar_dict[stock].mavg(15, frequency='day')
        # cur_price = bar_dict[stock].last

        avg_15 = h['close'][-15:].mean()
        cur_price = get_close_price(stock, 1, '1m')

        score = (cur_price - low_price_130) + (cur_price - high_price_130) + (cur_price - avg_15)
        # score = ((cur_price-low_price_130) + (cur_price-high_price_130) + (cur_price-avg_15)) / cur_price
        dst_stocks[stock] = score

    df = pd.Series(dst_stocks)
    df = df.sort_values()
    return df.index
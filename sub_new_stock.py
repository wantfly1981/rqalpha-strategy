
'''选择规则组合成一个量化策略'''

def select_strategy():
    '''
    策略选择设置说明:
    策略由以下步骤规则组合，组合而成:
    1.持仓股票的处理规则
    2.调仓条件判断规则
    3.Query选股规则 
    4.股票池过滤规则
    5.调仓规则
    6.其它规则

    '''
    # 规则配置list下标描述变量。提高可读性与未来添加更多规则配置。
    # 0.是否启用，1.描述，2.规则实现类名，3.规则传递参数(dict)]
    g.cs_enabled, g.cs_memo, g.cs_class_name, g.cs_param = range(4)

    # 配置 1.持仓股票的处理规则 (这里主要配置是否进行个股止损止盈)
    g.position_stock_config = [
        [True, '个股止损By ATR', 'Stop_loss_stocks_by_ATR', {}, ],
        [True, '比例止损', 'Stop_loss_stocks_by_percentage', {
            'percentage': 0.1,
        }],
    ]

    # 配置 2.调仓条件判断规则
    g.adjust_condition_config = [
        [True, '次新择时', 'Index_MACD_condition', {
            'index': '399678.XSHE',  # 次新指数
        }],
    ]

    # 配置 3.Query选股规则
    g.pick_stock_by_query_config = [
        [True, '选取流通小市值', 'Pick_small_circulating_market_cap', {}],
        [True, '过滤流通市值', 'Filter_circulating_market_cap', {
            'cm_cap_min': 0  # 最小流通市值
            , 'cm_cap_max': 2000000000  # 最大流通市值
        }],
        [True, '初选股票数量', 'Filter_limite', {
            'pick_stock_count': 200  # 备选股票数目
        }]
    ]

    # 配置 4.股票池过滤规则1（不需要处理数据）
    g.filter_stock_list_config = [
        [True, '过滤停牌', 'Filter_paused_stock', {}],
        [True, '过滤创业', 'Filter_gem', {}],
        [True, '过滤上市超过一定时间的股票', 'Filter_old_stock', {
            'day_count_max': 80,
            'day_count_min': 20
        }],
        [True, '过滤涨停板', 'Filter_limitup', {}],
        [True, '获取最终选股数', 'Filter_buy_count', {
            'buy_count': 6
        }],
    ]

    # 配置 4.1 股票池过滤规则2
    g.filter_stock_list_config_2 = [
        [True, '打分系统', 'Filter_score', {
            'rank_stock_count': 6
        }],
    ]

    # 配置 5.调仓规则
    g.adjust_position_config = [
        [True, '卖出规则', 'Sell_stocks', {}],
        [False, '买入股票', 'Buy_stocks_low', {
            'buy_count': 4  # 最终买入股票数
        }],
        [True, '买入股票', 'Buy_stocks_position', {
            'buy_count': 4  # 最终买入股票数
        }]
    ]

    g.handle_data_config = [
        [True, '处理分钟线数据', 'Handle_data_df', {}]
    ]

    # 配置 6.其它规则
    g.other_config = [
        [False, '统计', 'Stat', {}]
    ]


# 创建一个规则执行器，并初始化一些通用事件
def create_rule(class_name, params):
    '''
    在这借用eval函数，把规则配置里的字符串类名实例化。
    eval函数说明：将字符串当成有效Python表达式来求值，并返回计算结果
    x = 1
    y = eval('x+1')
    则结果为 y==2
    '''
    obj = eval(class_name)(params)
    obj.open_position_by_percent = open_position_by_percent
    obj.on_open_position = open_position
    obj.on_close_position = close_position
    obj.on_clear_position = clear_position
    return obj


# 根据规则配置创建规则执行器
def create_rules(config):
    # config里 0.是否启用，1.描述，2.规则实现类名，3.规则传递参数(dict)]
    return [create_rule(c[g.cs_class_name], c[g.cs_param]) for c in config if c[g.cs_enabled]]


def init(context):
    context.position = 0
    context.index_df = None
    context.index_df_15 = None
    context.index_df_30 = None
    context.index_df_60 = None
    context.stock_count = 0
    context.today = None
    context.stock_list = None
    context.result_df = None
    context.stock_to_buy = None
    # for ATR
    context.maxvalue = pd.DataFrame()
    context.ATRperiod = 22
    context.ATRList = []

    context.bar_5 = {}
    context.bar_15 = {}
    context.bar_30 = {}
    context.bar_60 = {}
    context.bar_1 = {}
    context.bar_day = {}

    context.stock_60 = []
    context.stock_30 = []
    context.stock_15 = []

    context.timedelt = 0
    context.score_df = {}
    context.black_list = []

    select_strategy()
    '''-----1.持仓股票的处理规则:-----'''
    g.position_stock_rules = create_rules(g.position_stock_config)

    '''-----2.调仓条件判断规则:-----'''
    g.adjust_condition_rules = create_rules(g.adjust_condition_config)

    '''-----3 数据处理规则:数据处理器-----'''
    g.handle_data_rules = create_rules(g.handle_data_config)

    '''-----4.Query选股规则:-----'''
    g.pick_stock_by_query_rules = create_rules(g.pick_stock_by_query_config)

    '''-----5.股票池过滤规则:-----'''
    g.filter_stock_list_rules = create_rules(g.filter_stock_list_config)
    g.filter_stock_list_rules_2 = create_rules(g.filter_stock_list_config_2)

    '''-----6.调仓规则:器-----'''
    g.adjust_position_rules = create_rules(g.adjust_position_config)

    '''-----7.其它规则:-------'''
    g.other_rules = create_rules(g.other_config)

    # 把所有规则合并排重生成一个总的规则收录器。以方便各处共同调用的
    g.all_rules = list(set(g.position_stock_rules
                           + g.adjust_condition_rules
                           + g.filter_stock_list_rules
                           + g.filter_stock_list_rules_2
                           + g.pick_stock_by_query_rules
                           + g.handle_data_rules
                           + g.adjust_position_rules
                           + g.other_rules
                           ))

    for rule in g.all_rules:
        rule.initialize(context)


def before_trading(context):
    logger.info("==================before_trading===========================================")

    # 盘前参数初始化
    context.bar_day = {}
    context.black_list = []

    # 基本面选股条件
    q = None
    data = None
    for rule in g.pick_stock_by_query_rules:
        q = rule.filter(context, data, q)

    # 基本面选股查询
    context.stock_list = list(get_fundamentals(q).columns.values)

    # 盘前选股处理_1
    for rule in g.filter_stock_list_rules:
        rule.before_trading_start(context)

    # 获取数据
    for rule in g.handle_data_rules:
        rule.before_trading_start(context)

    # 盘前选股处理_2
    for rule in g.filter_stock_list_rules_2:
        rule.before_trading_start(context)


def handle_bar(context, bar_dict):
    plot('actual postion', context.portfolio.market_value / context.portfolio.portfolio_value)
    plot('context.position', context.position)

    data = bar_dict
    # 分钟计算
    if context.now.hour < 12:
        context.timedelt = (context.now.hour - 9) * 60 + (context.now.minute - 30)
    else:
        context.timedelt = (context.now.hour - 9) * 60 + (context.now.minute - 30) - 90

    # 数据处理
    for rule in g.handle_data_rules:
        rule.handle_data(context, data)

    # 持仓股票动作的执行,目前为个股止损止盈
    if context.timedelt % 5 == 0:
        for rule in g.position_stock_rules:
            rule.handle_data(context, data)

    # -----------------------------------------------------------

    # 判断是否满足调仓条件，所有规则以and 逻辑执行
    for rule in g.adjust_condition_rules:
        rule.handle_data(context, data)

        if not rule.can_adjust:
            return

    # 执行调仓
    for rule in g.adjust_position_rules:
        rule.adjust(context, data, context.stock_list)

        # ----------------------------------------------------


def after_trading(context):
    '''
    for rule in g.all_rules:
        rule.after_trading_end(context)

    #基本面选股查询
    context.stock_list = list(get_fundamentals(q).columns.values)

    print('基本面选股 total_count = ', len(context.stock_list))

    #条件筛选股票
    for rule in g.filter_stock_list_rules:
        rule.before_trading_start(context)

    #股票指标打分
    for rule in g.filter_stock_score_rules:
        rule.before_trading_start(context)


    # 得到当前未完成订单
    orders = get_open_orders()
    for _order in orders.values():
        logger.info("canceled uncompleted order: %s" %(_order.order_id))
    '''


def log_param():
    def get_rules_str(rules):
        return '\n'.join(['   %d.%s ' % (i + 1, str(r)) for i, r in enumerate(rules)]) + '\n'

    s = '\n---------------------策略一览：规则组合与参数----------------------------\n'
    s += '一、持仓股票的处理规则:\n' + get_rules_str(g.position_stock_rules)
    s += '二、调仓条件判断规则:\n' + get_rules_str(g.adjust_condition_rules)
    s += '三、Query选股规则:\n' + get_rules_str(g.pick_stock_by_query_rules)
    s += '四、股票池过滤规则:\n' + get_rules_str(g.filter_stock_list_rules)
    s += '五、调仓规则:\n' + get_rules_str(g.adjust_position_rules)
    s += '五.5、处理规则:\n' + get_rules_str(g.handle_data_rules)
    s += '六、其它规则:\n' + get_rules_str(g.other_rules)
    s += '--------------------------------------------------------------------------'
    print(s)

''' ==============================规则基类================================'''
class Rule(object):
    # 每个子类必需写__name__,以修改策略时，方便判断规则器是否存在。
    __name__ = 'Base'
    # 持仓操作的事件
    on_open_position = None
    on_close_position = None
    on_clear_positions = None

    def __init__(self, params):
        pass

    def initialize(self, context):
        pass

    def handle_data(self, context, data):
        pass

    def before_trading_start(self, context):
        pass

    def after_trading_end(self, context):
        pass

    def process_initialize(self, context):
        pass

    def after_code_changed(self, context):
        pass

    # 卖出股票时调用的函数
    # price为当前价，amount为发生的股票数,is_normal正常规则卖出为True，止损卖出为False
    def when_sell_stock(self, position, order, is_normal):
        pass

    # 买入股票时调用的函数
    # price为当前价，amount为发生的股票数
    def when_buy_stock(self, stock, order):
        pass

    # 清仓时调用的函数
    def when_clear_position(self, context):
        pass

    # 调仓前调用
    def before_adjust_start(self, context, data):
        pass

    # 调仓后调用用
    def after_adjust_end(slef, context, data):
        pass

    # 更改参数
    def update_params(self, context, params):
        pass

    # 持仓操作事件的简单判断处理，方便使用。
    def open_position(self, security, value):
        if self.on_open_position != None:
            self.on_open_position(security, value)

    def close_position(self, position, is_normal=True):
        if self.on_close_position != None:
            self.on_close_position(position, is_normal=True)

    def clear_positions(self, context):
        if self.on_clear_positions != None:
            self.on_clear_positions(context)

# 一下内容可另存为文件 adjust_condition.py
# -*- coding:utf-8 -*-

# from rule import *
# from util import *
# import pandas as pd

'''==============================调仓条件判断器基类=============================='''


class Adjust_condition(Rule):
    __name__ = 'Adjust_condition'

    # 返回能否进行调仓
    @property
    def can_adjust(self):
        return True


''' ----------------------最高价最低价比例止损------------------------------'''


class Stop_loss_by_price(Adjust_condition):
    __name__ = 'Stop_loss_by_price'

    def __init__(self, params):
        self.index = params.get('index', '000001.XSHG')
        self.day_count = params.get('day_count', 160)
        self.multiple = params.get('multiple', 2.2)
        self.is_day_stop_loss_by_price = False

    def update_params(self, context, params):
        self.index = params.get('index', self.index)
        self.day_count = params.get('day_count', self.day_count)
        self.multiple = params.get('multiple', self.multiple)

    def handle_data(self, context, data):
        # 大盘指数前130日内最高价超过最低价2倍，则清仓止损
        # 基于历史数据判定，因此若状态满足，则当天都不会变化
        # 增加此止损，回撤降低，收益降低

        if not self.is_day_stop_loss_by_price:
            h = history_bars(self.index, self.day_count, '1d', ('close', 'high', 'low'))
            low_price_130 = h.low.min()
            high_price_130 = h.high.max()
            if high_price_130 > self.multiple * low_price_130 and h['close'][-1] < h['close'][-4] * 1 and h['close'][
                -1] > h['close'][-100]:
                # 当日第一次输出日志
                logger.info("==> 大盘止损，%s指数前130日内最高价超过最低价2倍, 最高价: %f, 最低价: %f" % (
                get_security_info(self.index).display_name, high_price_130, low_price_130))
                self.is_day_stop_loss_by_price = True

        if self.is_day_stop_loss_by_price:
            self.clear_position(context)

    def before_trading_start(self, context):
        self.is_day_stop_loss_by_price = False
        pass

    def __str__(self):
        return '大盘高低价比例止损器:[指数: %s] [参数: %s日内最高最低价: %s倍] [当前状态: %s]' % (
            self.index, self.day_count, self.multiple, self.is_day_stop_loss_by_price)

    @property
    def can_adjust(self):
        return not self.is_day_stop_loss_by_price


''' ----------------------三乌鸦止损------------------------------'''


class Stop_loss_by_3_black_crows(Adjust_condition):
    __name__ = 'Stop_loss_by_3_black_crows'

    def __init__(self, params):
        self.index = params.get('index', '000001.XSHG')
        self.dst_drop_minute_count = params.get('dst_drop_minute_count', 60)
        # 临时参数
        self.is_last_day_3_black_crows = False
        self.t_can_adjust = True
        self.cur_drop_minute_count = 0

    def update_params(self, context, params):
        self.index = params.get('index', self.index)
        self.dst_drop_minute_count = params.get('dst_drop_minute_count', self.dst_drop_minute_count)

    def initialize(self, context):
        pass

    def handle_data(self, context, data):
        # 前日三黑鸦，累计当日每分钟涨幅<0的分钟计数
        # 如果分钟计数超过一定值，则开始进行三黑鸦止损
        # 避免无效三黑鸦乱止损
        if self.is_last_day_3_black_crows:
            if get_growth_rate(self.index, 1) < 0:
                self.cur_drop_minute_count += 1

            if self.cur_drop_minute_count >= self.dst_drop_minute_count:
                if self.cur_drop_minute_count == self.dst_drop_minute_count:
                    logger.info("==> 超过三黑鸦止损开始")

                self.clear_position(context)
                self.t_can_adjust = False
        else:
            self.t_can_adjust = True
        pass

    def before_trading_start(self, context):
        self.is_last_day_3_black_crows = is_3_black_crows(self.index)
        if self.is_last_day_3_black_crows:
            logger.info("==> 前4日已经构成三黑鸦形态")
        pass

    def after_trading_end(self, context):
        self.is_last_day_3_black_crows = False
        self.cur_drop_minute_count = 0
        pass

    def __str__(self):
        return '大盘三乌鸦止损器:[指数: %s] [跌计数分钟: %d] [当前状态: %s]' % (
            self.index, self.dst_drop_minute_count, self.is_last_day_3_black_crows)

    @property
    def can_adjust(self):
        return self.t_can_adjust


''' ----------------------28指数值实时进行止损------------------------------'''


class Stop_loss_by_28_index(Adjust_condition):
    __name__ = 'Stop_loss_by_28_index'

    def __init__(self, params):
        self.index2 = params.get('index2', '')
        self.index8 = params.get('index8', '')
        self.index_growth_rate = params.get('index_growth_rate', 0.01)
        self.dst_minute_count_28index_drop = params.get('dst_minute_count_28index_drop', 120)
        # 临时参数
        self.t_can_adjust = True
        self.minute_count_28index_drop = 0

    def update_params(self, context, params):
        self.index2 = params.get('index2', self.index2)
        self.index8 = params.get('index8', self.index8)
        self.index_growth_rate = params.get('index_growth_rate', self.index_growth_rate)
        self.dst_minute_count_28index_drop = params.get('dst_minute_count_28index_drop',
                                                        self.dst_minute_count_28index_drop)

    def initialize(self, context):
        pass

    def handle_data(self, context, data):
        # 回看指数前20天的涨幅
        gr_index2 = get_growth_rate(self.index2)
        gr_index8 = get_growth_rate(self.index8)

        if gr_index2 <= self.index_growth_rate and gr_index8 <= self.index_growth_rate:
            if (self.minute_count_28index_drop == 0):
                logger.info("当前二八指数的20日涨幅同时低于[%.2f%%], %s指数: [%.2f%%], %s指数: [%.2f%%]" \
                            % (self.index_growth_rate * 100,
                               get_security_info(self.index2).display_name,
                               gr_index2 * 100,
                               get_security_info(self.index8).display_name,
                               gr_index8 * 100))

            self.minute_count_28index_drop += 1
        else:
            # 不连续状态归零
            if self.minute_count_28index_drop < self.dst_minute_count_28index_drop:
                self.minute_count_28index_drop = 0

        if self.minute_count_28index_drop >= self.dst_minute_count_28index_drop:
            if self.minute_count_28index_drop == self.dst_minute_count_28index_drop:
                logger.info("==> 当日%s指数和%s指数的20日增幅低于[%.2f%%]已超过%d分钟，执行28指数止损" \
                            % (get_security_info(self.index2).display_name, get_security_info(self.index8).display_name,
                               self.index_growth_rate * 100, self.dst_minute_count_28index_drop))

            self.clear_position(context)
            self.t_can_adjust = False
        else:
            self.t_can_adjust = True
        pass

    def after_trading_end(self, context):
        self.t_can_adjust = False
        self.minute_count_28index_drop = 0
        pass

    def __str__(self):
        return '28指数值实时进行止损:[大盘指数: %s %s] [小盘指数: %s %s] [判定调仓的二八指数20日增幅 %.2f%%]' % (
            self.index2, self.index8, self.index_growth_rate * 100)

    @property
    def can_adjust(self):
        return self.t_can_adjust


'''-------------------------调仓时间控制器-----------------------'''


class Time_condition(Adjust_condition):
    __name__ = 'Time_condition'

    def __init__(self, params):
        # 配置调仓时间（24小时分钟制）
        self.hour = params.get('hour', 14)
        self.minute = params.get('minute', 50)

    def update_params(self, context, params):
        self.hour = params.get('hour', self.hour)
        self.minute = params.get('minute', self.minute)
        pass

    @property
    def can_adjust(self):
        return self.t_can_adjust

    def handle_data(self, context, data):
        hour = context.now.hour
        minute = context.now.minute
        self.t_can_adjust = hour >= self.hour and minute <= self.minute + 10
        pass

    def __str__(self):
        return '调仓时间控制器: [调仓时间: %d:%d]' % (
            self.hour, self.minute)


'''-------------------------调仓日计数器-----------------------'''


class Period_condition(Adjust_condition):
    __name__ = 'Period_condition'

    def __init__(self, params):
        # 调仓日计数器，单位：日
        self.period = params.get('period', 3)
        self.day_count = 0
        self.t_can_adjust = False

    def update_params(self, context, params):
        self.period = params.get('period', self.period)

    @property
    def can_adjust(self):
        return self.t_can_adjust

    # todo:分钟线回测
    def handle_data(self, context, data):
        print("调仓日计数 [%d]" % (self.day_count))
        self.t_can_adjust = self.day_count % self.period == 0

        print(context.today, context.now.date())

        if context.today != context.now.date():
            context.today = context.now.date()
            self.day_count += 1
        pass

    def before_trading_start(self, context):
        self.t_can_adjust = False
        pass

    def when_sell_stock(self, position, order, is_normal):
        if not is_normal:
            # 个股止损止盈时，即非正常卖股时，重置计数，原策略是这么写的
            self.day_count = 0
        pass

    # 清仓时调用的函数
    def when_clear_position(self, context):
        self.day_count = 0
        pass

    def __str__(self):
        return '调仓日计数器:[调仓频率: %d日] [调仓日计数 %d]' % (
            self.period, self.day_count)


'''-------------------------28指数涨幅调仓判断器----------------------'''


class Index28_condition(Adjust_condition):
    __name__ = 'Index28_condition'

    def __init__(self, params):
        self.index2 = params.get('index2', '')
        self.index8 = params.get('index8', '')
        self.index_growth_rate = params.get('index_growth_rate', 0.01)
        self.t_can_adjust = False

    def update_params(self, context, params):
        self.index2 = params.get('index2', self.index2)
        self.index8 = params.get('index8', self.index8)
        self.index_growth_rate = params.get('index_growth_rate', self.index_growth_rate)

    @property
    def can_adjust(self):
        return self.t_can_adjust

    def handle_data(self, context, data):
        # 回看指数前20天的涨幅
        gr_index2 = get_growth_rate(self.index2)
        gr_index8 = get_growth_rate(self.index8)

        if gr_index2 <= self.index_growth_rate and gr_index8 <= self.index_growth_rate:
            self.clear_position(context)
            self.t_can_adjust = False
        else:
            self.t_can_adjust = True
        pass

    def before_trading_start(self, context):
        pass

    def __str__(self):
        return '28指数择时:[大盘指数:%s %s] [小盘指数:%s %s] [判定调仓的二八指数20日增幅 %.2f%%]' % (
            self.index2, instruments(self.index2).symbol,
            self.index8, instruments(self.index8).symbol,
            self.index_growth_rate * 100)


'''-------------------------当日大跌调仓判断器----------------------'''


class BigLost_condition(Adjust_condition):
    __name__ = 'BigLost_condition'

    def __init__(self, params):
        self.index = params.get('index', '')
        self.t_can_adjust = False

    def update_params(self, context, params):
        self.index = params.get('index', self.index)

    @property
    def can_adjust(self):
        return self.t_can_adjust

    def handle_data(self, context, data):
        snapshot_index = current_snapshot(self.index)
        # logger.info("当前指数的跌幅 [%.2f%%]" %(1 - (snapshot_index.last / snapshot_index.prev_close)))

        if (snapshot_index.last / snapshot_index.prev_close) < (1 - 0.03):
            self.clear_position(context)
            self.t_can_adjust = False
        else:
            self.t_can_adjust = True
        pass

    def before_trading_start(self, context):
        pass

    def __str__(self):
        return '当前指数的跌幅调仓'


'''-------------------------MACD指数涨幅调仓判断器----------------------'''


class Index_MACD_condition(Adjust_condition):
    __name__ = 'Index_MACD_condition'

    def __init__(self, params):
        self.index = params.get('index', '')
        self.t_can_adjust = False

    def update_params(self, context, params):
        self.index = params.get('index', self.index)

    @property
    def can_adjust(self):
        return self.t_can_adjust

    def handle_data(self, context, data):

        context.position = 0

        # 日线
        if context.index_df.iloc[-1]['macd'] > 0:
            context.position = 1
        else:
            context.position = 0.5

        if context.position > 0:
            self.t_can_adjust = True
        else:
            self.clear_position(context)
            self.t_can_adjust = False

        return self.t_can_adjust

    def before_trading_start(self, context):

        pass

    '''
    def __str__(self):
        return '28指数择时:[大盘指数:%s %s] [小盘指数:%s %s] [判定调仓的二八指数20日增幅 %.2f%%]'%(
                self.index2,instruments(self.index2).symbol,
                self.index8,instruments(self.index8).symbol,
                self.index_growth_rate * 100)
    '''


# 一下内容可另存为文件 adjust_position.py

# -*- coding:utf-8 -*-

# from rule import *
# from util import *


'''==============================个股止盈止损规则=============================='''

''' ---------个股止损 by 自最高值回落一定比例比例进行止损-------------------------'''


class Stop_loss_stocks_by_percentage(Rule):
    __name__ = 'Stop_loss_stocks_by_percentage'

    def __init__(self, params):
        self.percent = params.get('percent', 0.08)

    def update_params(self, context, params):
        self.percent = params.get('percent', self.period)

    # 个股止损
    def handle_data(self, context, data):

        # 持仓股票循环
        for stock in context.portfolio.positions.keys():

            # 持有数量超过0
            if context.portfolio.positions[stock].quantity > 0:

                # 当前价格
                cur_price = data[stock].close

                # 历史最高价格
                stockdic = context.maxvalue[stock]
                highest = stockdic[0]

                if data[stock].high > highest:
                    del context.maxvalue[stock]
                    temp = pd.DataFrame({str(stock): [max(highest, data[stock].high)]})
                    context.maxvalue = pd.concat([context.maxvalue, temp], axis=1, join='inner')  # 更新其盘中最高价值和先阶段比例。

                    # 更新历史最高价格
                    stockdic = context.maxvalue[stock]
                    highest = stockdic[0]

                if cur_price < highest * (1 - self.percent):
                    position = context.portfolio.positions[stock]
                    self.close_position(position, False)
                    context.black_list.append(stock)

                    print('[比例止损卖出]', instruments(stock).symbol, context.portfolio.positions[stock].avg_price, highest,
                          data[stock].last)
            else:
                if stock in context.ATRList:
                    context.ATRList.remove(stock)

                    # if stock in context.maxvalue.keys:
                    # del context.maxvalue[stock]

    def when_sell_stock(self, position, order, is_normal):
        # if position.security in self.last_high:
        #    self.last_high.pop(position.security)
        pass

    def when_buy_stock(self, stock, order):
        # if order.status == OrderStatus.held and order.filled == order.amount:
        # 全部成交则删除相关证券的最高价缓存
        #    self.last_high[stock] = get_close_price(stock, 1, '1m')
        pass

    def __str__(self):
        return '个股止损器:[按比例止损: %d ]' % self.percent


''' ----------------------个股止损 by ATR 60-------------------------------------'''


class Stop_loss_stocks_by_ATR(Rule):
    __name__ = 'Stop_loss_stocks_by_ATR'

    def __init__(self, params):
        pass

    def update_params(self, context, params):
        pass

    # 个股止损
    def handle_data(self, context, data):

        for stock in context.ATRList:

            if stock in context.portfolio.positions.keys() and context.portfolio.positions[stock].quantity > 0:

                # 当前涨幅判断
                raisePercentage = (data[stock].close - context.portfolio.positions[stock].avg_price) / \
                                  context.portfolio.positions[stock].avg_price

                if raisePercentage > 0.12:

                    print(context.bar_60[stock])
                    bar = context.bar_60

                else:
                    if raisePercentage > 0.8:

                        bar = context.bar_30

                    else:
                        if raisePercentage > 0.4:

                            bar = context.bar_15

                        else:
                            return

                ATR = findATR(context, bar, stock)

                high = bar[stock].iloc[-1]['high']
                current = bar[stock].iloc[-1]['close']
                stockdic = context.maxvalue[stock]
                highest = stockdic[0]

                del context.maxvalue[stock]
                temp = pd.DataFrame({str(stock): [max(highest, high)]})

                context.maxvalue = pd.concat([context.maxvalue, temp], axis=1, join='inner')  # 更新其盘中最高价值和先阶段比例。

                stockdic = context.maxvalue[stock]
                highest = stockdic[0]

                if data[stock].close < highest - 3 * ATR:
                    print('[ATR止损卖出]', instruments(stock).symbol, context.portfolio.positions[stock].avg_price, highest,
                          data[stock].last, ATR)
                    position = context.portfolio.positions[stock]
                    self.close_position(position)
                    context.black_list.append(stock)
            else:
                del context.maxvalue[stock]
                context.ATRList.remove(stock)

    def when_sell_stock(self, position, order, is_normal):
        # if position.security in self.last_high:
        #    self.last_high.pop(position.security)
        pass

    def when_buy_stock(self, stock, order):
        # if order.status == OrderStatus.held and order.filled == order.amount:
        # 全部成交则删除相关证券的最高价缓存
        #    self.last_high[stock] = get_close_price(stock, 1, '1m')
        pass

    def after_trading_end(self, context):
        # self.pct_change = {}
        pass

    def __str__(self):
        return '个股止损器:ATR止损'


'''==============================调仓的操作基类================================'''


class Adjust_position(Rule):
    __name__ = 'Adjust_position'

    def adjust(self, context, data, buy_stocks):
        pass


'''---------------卖出股票规则------------------------'''
'''---------------个股涨幅超过5%，进入ATR--------------'''


class Sell_stocks(Adjust_position):
    __name__ = 'Sell_stocks'

    def adjust(self, context, data, buy_stocks):

        for stock in context.portfolio.positions.keys():

            if context.portfolio.positions[stock].quantity == 0:
                return

            if context.portfolio.positions[stock].sellable == 0:
                return

            if data[stock].close < context.portfolio.positions[stock].avg_price * 1.04:

                # 止损
                if data[stock].close < context.portfolio.positions[stock].avg_price * 0.92:
                    position = context.portfolio.positions[stock]
                    self.close_position(position)
                    context.black_list.append(stock)

            else:
                if stock not in context.ATRList:  # and data[stock].close > context.portfolio.positions[stock].avg_price * 1.04:
                    context.ATRList.append(stock)

            if stock in context.stock_60:
                # 涨幅 8%
                if data[stock].close > context.portfolio.positions[stock].avg_price * 1.07:
                    positions = context.portfolio.positions[stock]
                    percentage = context.portfolio.positions[stock].value_percent

                    print(stock, instruments(stock).symbol, data[stock].close, '60分钟 7%止盈卖出')
                    close_position_2(positions, percentage / 2)
                    context.black_list.append(stock)
                    context.stock_60.remove(stock)

            if stock in context.stock_30:
                # 涨幅 4%
                if data[stock].close > context.portfolio.positions[stock].avg_price * 1.04:
                    positions = context.portfolio.positions[stock]
                    percentage = context.portfolio.positions[stock].value_percent

                    print(stock, instruments(stock).symbol, data[stock].close, '30分钟 4%止盈卖出')
                    close_position_2(positions, percentage / 2)
                    context.black_list.append(stock)
                    context.stock_30.remove(stock)

            if stock in context.stock_15:
                # 涨幅 2.5%
                if data[stock].close > context.portfolio.positions[stock].avg_price * 1.025:
                    positions = context.portfolio.positions[stock]
                    percentage = context.portfolio.positions[stock].value_percent

                    print(stock, instruments(stock).symbol, data[stock].close, '15分钟 2.5%止盈卖出')
                    close_position_2(positions, percentage / 2)
                    context.black_list.append(stock)
                    context.stock_15.remove(stock)

    def __str__(self):
        return '股票调仓卖出规则：卖出不在buy_stocks的股票'


'''---------------买入股票规则 补足仓位--------------'''


class Buy_stocks_position(Adjust_position):
    __name__ = 'Buy_stocks'

    def __init__(self, params):
        self.buy_count = params.get('buy_count', 4)

    def update_params(self, context, params):
        self.buy_count = params.get('buy_count', self.buy_count)

    def adjust(self, context, data, buy_stocks):

        # 买入股票 交易时间为下午2点
        if (context.timedelt != 180):
            return

        if context.index_df.iloc[-1]['macd'] < 0:
            return

        actual_position = context.portfolio.market_value / context.portfolio.portfolio_value

        if actual_position > context.position * 0.95:
            return

        # 避免小额下单
        if context.portfolio.cash < 10000:
            return

        buy_stock_list = []

        stock_list_count = len(context.stock_list)

        for stock in context.stock_list:
            if stock in context.black_list:
                return

            createdic(context, data, stock)
            if context.portfolio.positions[stock].value_percent * 1.05 < (1 / self.buy_count):
                self.open_position_by_percent(stock, (1 / self.buy_count))
                print('[补仓买入]', instruments(stock).symbol, (1 / self.buy_count))

        pass

    def __str__(self):
        return '股票调仓买入规则：现金平分式买入股票达目标股票数'


class Buy_stocks2(Adjust_position):
    __name__ = 'Buy_stocks2'

    def __init__(self, params):
        self.buy_count = params.get('buy_count', 3)
        # self.buy_position = params.get('buy_position', 0)

    def update_params(self, context, params):
        self.buy_count = params.get('buy_count', self.buy_count)
        # self.buy_position = params.get('buy_position', self.buy_position)

    def adjust(self, context, data, buy_stocks):
        # 买入股票
        # 始终保持持仓数目为g.buy_stock_count
        # 根据股票数量分仓
        # 此处只根据可用金额平均分配购买，不能保证每个仓位平均分配
        print(context.portfolio.cash, context.portfolio.market_value, context.portfolio.portfolio_value,
              context.position)

        if (context.portfolio.market_value / context.portfolio.portfolio_value) < context.position:
            self.buy_position = context.position - context.portfolio.market_value / context.portfolio.portfolio_value
            value = context.portfolio.portfolio_value * self.buy_position - context.portfolio.market_value
            buy_value = value / self.buy_count
            for stock in buy_stocks:
                self.open_position(stock, buy_value)

        pass

    def __str__(self):
        return '股票调仓买入规则：现金平分式买入股票达目标股票数'

'''==============================选股 query过滤器基类=============================='''

class Filter_query(Rule):
    __name__ = 'Filter_query'

    def filter(self, context, data, q):
        return None


'''------------------小市值选股器-----------------'''


class Pick_small_cap(Filter_query):
    __name__ = 'Pick_small_cap'

    def filter(self, context, data, q):
        return query(
            fundamentals.eod_derivative_indicator.market_cap
        ).order_by(
            fundamentals.eod_derivative_indicator.market_cap.asc()
        )

    def __str__(self):
        return '按总市值倒序选取股票'


class Pick_small_circulating_market_cap(Filter_query):
    __name__ = 'Pick_small_circulating_market_cap'

    def filter(self, context, data, q):
        return query(
            fundamentals.eod_derivative_indicator.a_share_market_val_2
        ).order_by(
            fundamentals.eod_derivative_indicator.a_share_market_val_2.asc()
        )

    def __str__(self):
        return '按流通市值倒序选取股票'


class Filter_pe(Filter_query):
    __name__ = 'Filter_pe'

    def __init__(self, params):
        self.pe_min = params.get('pe_min', 0)
        self.pe_max = params.get('pe_max', 200)

    def update_params(self, context, params):
        self.pe_min = params.get('pe_min', self.pe_min)
        self.pe_max = params.get('pe_max', self.pe_max)

    def filter(self, context, data, q):
        return q.filter(
            fundamentals.eod_derivative_indicator.pe_ratio_2 > self.pe_min,
            fundamentals.eod_derivative_indicator.pe_ratio_2 < self.pe_max
        )

    def __str__(self):
        return '根据动态PE范围选取股票： [ %d < pe < %d]' % (self.pe_min, self.pe_max)


class Filter_circulating_market_cap(Filter_query):
    __name__ = 'Filter_market_cap'

    def __init__(self, params):
        self.cm_cap_min = params.get('cm_cap_min', 0)
        self.cm_cap_max = params.get('cm_cap_max', 10000000000)

    def update_params(self, context, params):
        self.cm_cap_min = params.get('cm_cap_min', self.cm_cap_min)
        self.cm_cap_max = params.get('cm_cap_max', self.cm_cap_max)

    def filter(self, context, data, q):
        return q.filter(
            fundamentals.eod_derivative_indicator.a_share_market_val_2 <= self.cm_cap_max,
            fundamentals.eod_derivative_indicator.a_share_market_val_2 >= self.cm_cap_min
        )

    def __str__(self):
        return '根据流通市值范围选取股票： [ %d < circulating_market_cap < %d]' % (self.cm_cap_min, self.cm_cap_max)


class Filter_limite(Filter_query):
    __name__ = 'Filter_limite'

    def __init__(self, params):
        self.pick_stock_count = params.get('pick_stock_count', 0)

    def update_params(self, context, params):
        self.pick_stock_count = params.get('pick_stock_count', self.pick_stock_count)

    def filter(self, context, data, q):
        return q.limit(self.pick_stock_count)

    def __str__(self):
        return '初选股票数量: %d' % (self.pick_stock_count)


# 一下内容可另存为 filter_stock_list.py

# -*- coding:utf-8 -*-

# from kuanke.user_space_api import *
from rule import *
from util import *
import pandas as pd
import numpy as np
import datetime

'''==============================选股 stock_list过滤器基类=============================='''


class Filter_stock_list(Rule):
    __name__ = 'Filter_stock_list'

    def before_trading_start(self, context):
        return None

    def filter(self, context, data, stock_list):
        return None


class Filter_gem(Filter_stock_list):
    __name__ = 'Filter_gem'

    def before_trading_start(self, context):

        result_list = []

        for stock in context.stock_list:
            if stock[0:3] != '300':
                result_list.append(stock)
                pass

        context.stock_list = result_list

        return None

    def filter(self, context, data, stock_list):
        return [stock for stock in stock_list if stock[0:3] != '300']

    def __str__(self):
        return '过滤创业板股票'


class Filter_paused_stock(Filter_stock_list):
    __name__ = 'Filter_paused_stock'

    def before_trading_start(self, context):

        result_list = []

        for stock in context.stock_list:
            if not is_suspended(stock):
                result_list.append(stock)
                pass

        context.stock_list = result_list

        return None

    def filter(self, context, data, stock_list):
        return [stock for stock in stock_list
                if not is_suspended(stock)
                ]

    def __str__(self):
        return '过滤停牌股票'


class Filter_limitup(Filter_stock_list):
    __name__ = 'Filter_limitup'

    def before_trading_start(self, context):

        # 盘前过滤前日涨停
        result_list = []

        for stock in context.stock_list:
            h = history_bars(stock, 1, '1d', 'high')
            l = history_bars(stock, 1, '1d', 'low')

            if h[0] != l[0] and instruments(stock).days_from_listed() > 5:
                result_list.append(stock)

        context.stock_list = result_list

        return context.stock_list

    def filter(self, context, data, stock_list):
        # todo: 2月14日的数据有问题?
        return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
                or data[stock].close < data[stock].limit_up or data[stock].high != data[stock].low]

    def __str__(self):
        return '过滤涨停股票,已持仓股不过滤'


'''
class Filter_drop_percentage(Filter_stock_list):
    __name__='Filter_drop_percentage'

    def __init__(self,params):
        self.drop_percent = params.get('drop_percent', -0.05)
    def update_params(self,context,params):
        self.drop_percent = params.get('drop_percent', drop_percent)

    def filter(self,context,data,stock_list):
        for stock in stock_list:
            if ((current_snapshot(stock).last - current_snapshot(stock).prev_close)/ current_snapshot(stock).prev_close) < self.drop_percent:
                print(stock, '过滤当日跌幅-5', current_snapshot(stock).last, current_snapshot(stock).prev_close)


        return [stock for stock in stock_list 
            if ((current_snapshot(stock).last - current_snapshot(stock).prev_close)/ current_snapshot(stock).prev_close) > self.drop_percent]

    def __str__(self):
        return '过滤当日跌幅5%的股票'
'''
'''     
class Filter_limitdown(Filter_stock_list):
    __name__='Filter_limitdown'
    def filter(self,context,data,stock_list):
        return [stock for stock in stock_list 
            if stock in context.portfolio.positions.keys()
            or data[stock].close > data[stock].limit_down]
    def __str__(self):
        return '过滤跌停股票,已持仓股不过滤' 
'''
'''
class Filter_growth_rate_stock(Filter_stock_list):
    __name__='Filter_growth_rate_stock'
    def __init__(self,params):
        self.gr_max = params.get('gr_max', 0.2)
        self.gr_min = params.get('gr_min', -0.05)
        self.period = params.get('period', 5)
    def before_trading_start(self, context):
        return context.stock_list
    def update_params(self,context,params):
        self.gr_max = params.get('gr_max',self.gr_max)
        self.gr_min = params.get('gr_min',self.gr_min)
        self.period = params.get('period',self.period)
    def filter(self,context,data,stock_list):
        return [stock for stock in stock_list 
            if get_growth_rate(stock, self.period) > self.gr_min and
            get_growth_rate(stock, self.period) < self.gr_max]
    def __str__(self):
        return '过滤n日涨幅为特定值的股票'
'''


class Filter_old_stock(Filter_stock_list):
    __name__ = 'Filter_old_stock'

    def __init__(self, params):
        self.day_count_min = params.get('day_count_min', 5)
        self.day_count_max = params.get('day_count_max', 80)

    def before_trading_start(self, context):
        context.stock_list = [stock for stock in context.stock_list
                              if instruments(stock).days_from_listed() <= self.day_count_max and instruments(
                stock).days_from_listed() >= self.day_count_min]
        return context.stock_list

    def update_params(self, context, params):
        self.day_count_min = params.get('day_count_min', self.day_count_min)
        self.day_count_max = params.get('day_count_max', self.day_count_max)

    def filter(self, context, data, stock_list):
        return [stock for stock in stock_list
                if instruments(stock).days_from_listed() <= self.day_count_max and instruments(
                stock).days_from_listed() >= self.day_count_min]

    def __str__(self):
        return '过滤上市时间超过 %d 天的股票' % (self.day_count)


'''
class Filter_turnover_rate_stock(Filter_stock_list):
    __name__='Filter_turnover_rate_stock'
    def __init__(self,params):
        self.turnover_rate_max = params.get('turnover_rate_max', 0.16)
        self.turnover_rate_min = params.get('turnover_rate_min', 0.03)
    def update_params(self,context,params):
        self.turnover_rate_max = params.get('turnover_rate_max',self.turnover_rate_max)
        self.turnover_rate_min = params.get('turnover_rate_min',self.turnover_rate_min)
    def filter(self,context,data,stock_list):
        return [stock for stock in stock_list 
            if get_turnover_rate(stock, count = 1).iloc[0, 0] >= self.turnover_rate_min 
            and self.turnover_rate_max >= get_turnover_rate(stock, count = 1).iloc[0, 0]
            ]
    def __str__(self):
        return '过滤换手率' 
'''
'''
class Filter_CS_stock(Filter_stock_list):
    __name__='Filter_cs_stock'
    def filter(self,context,data,stock_list):
        return [stock for stock in stock_list 
            if get_securities_margin(stock, count = 1)['margin_balance'][0] > 0]

    def __str__(self):
        return '过滤融资融券股票'
'''


class Filter_just_open_limit(Filter_stock_list):
    __name__ = '过滤新开板股票'

    def before_trading_start(self, context):
        for stock in context.stock_list:
            # 两种可能，前一天如果是涨停则为开板，前一天不是涨停，就是跌入股票池
            history = history_bars(stock, 2, '1d', 'close')
            if len(history) == 2 and history[0] * 1.099 > history[1]:
                result_list.append(stock)

        return context.stock_list

    def filter(self, context, data, stock_list):

        result_list = []

        if context.stock_list == None:
            context.stock_list = stock_list
            result_list = stock_list
        else:
            for stock in stock_list:
                if stock not in context.stock_list:
                    # 两种可能，前一天如果是涨停则为开板，前一天不是涨停，就是跌入股票池
                    history = history_bars(stock, 2, '1d', 'close')
                    if len(history) == 2 and history[0] * 1.099 > history[1]:
                        result_list.append(stock)
                else:
                    result_list.append(stock)

        return result_list

    def __str__(self):
        return '过滤新开板股票'


class Filter_st(Filter_stock_list):
    __name__ = 'Filter_st'

    def filter(self, context, data, stock_list):
        current_data = get_current_data()
        return [stock for stock in stock_list
                if not is_st_stock(stock)
                ]

    def __str__(self):
        return '过滤ST股票'


class Filter_growth_is_down(Filter_stock_list):
    __name__ = 'Filter_growth_is_down'

    def __init__(self, params):
        self.day_count = params.get('day_count', 20)

    def update_params(self, context, params):
        self.day_count = params.get('day_count', self.day_count)

    def filter(self, context, data, stock_list):
        return [stock for stock in stock_list if get_growth_rate(stock, self.day_count) > 0]

    def __str__(self):
        return '过滤n日增长率为负的股票'


class Filter_blacklist(Filter_stock_list):
    __name__ = 'Index28_condition'

    def __get_blacklist(self):
        # 黑名单一览表，更新时间 2016.7.10 by 沙米
        # 科恒股份、太空板业，一旦2016年继续亏损，直接面临暂停上市风险
        blacklist = ["600656.XSHG", "300372.XSHE", "600403.XSHG", "600421.XSHG", "600733.XSHG", "300399.XSHE",
                     "600145.XSHG", "002679.XSHE", "000020.XSHE", "002330.XSHE", "300117.XSHE", "300135.XSHE",
                     "002566.XSHE", "002119.XSHE", "300208.XSHE", "002237.XSHE", "002608.XSHE", "000691.XSHE",
                     "002694.XSHE", "002715.XSHE", "002211.XSHE", "000788.XSHE", "300380.XSHE", "300028.XSHE",
                     "000668.XSHE", "300033.XSHE", "300126.XSHE", "300340.XSHE", "300344.XSHE", "002473.XSHE"]
        return blacklist

    def filter(self, context, data, stock_list):
        blacklist = self.__get_blacklist()
        return [stock for stock in stock_list if stock not in blacklist]

    def __str__(self):
        return '过滤黑名单股票'


class Filter_score(Filter_stock_list):
    __name__ = 'Filter_score'

    def __init__(self, params):
        self.rank_stock_count = params.get('rank_stock_count', 20)

    def update_params(self, context, params):
        self.rank_stock_count = params.get('self.rank_stock_count', self.rank_stock_count)

    def before_trading_start(self, context):
        if len(context.stock_list) > self.rank_stock_count:
            context.stock_list = context.stock_list[:self.rank_stock_count]

        return None

    def filter(self, context, data, stock_list):
        return None

    def __str__(self):
        return '股票评分排序 [评分股数: %d ]' % (self.rank_stock_count)


class Filter_buy_count(Filter_stock_list):
    __name__ = 'Filter_buy_count'

    def __init__(self, params):
        self.buy_count = params.get('buy_count', 3)

    def update_params(self, context, params):
        self.buy_count = params.get('buy_count', self.buy_count)

    def before_trading_start(self, context):

        if len(context.stock_list) > self.buy_count:
            context.stock_list = context.stock_list[:self.buy_count]

        return context.stock_list

    def filter(self, context, data, stock_list):

        if len(context.stock_list) > self.buy_count:
            context.stock_list = context.stock_list[:self.buy_count]

        return context.stock_list

    def __str__(self):
        return '获取最终待购买股票数:[ %d ]' % (self.buy_count)


# 一下内容可另存为 handle_data_rule

# -*- coding:utf-8 -*-

# from kuanke.user_space_api import *
# from rule import *
# from util import *
# import pandas as pd

'''==============================选股 stock_list过滤器基类=============================='''


class Handle_data_rule(Rule):
    __name__ = 'Handle_data_rule'

    def before_trading_start(self, context):
        return None

    def handle_data(self, context, data):
        return None

    def after_trading_end(self, context):
        return None


class Handle_data_df(Handle_data_rule):
    __name__ = 'Handle_data_df'

    def before_trading_start(self, context):

        today = context.now.date();

        # 补全数据
        # 选股池列表
        for stock in context.stock_list:
            context.bar_60[stock] = get_price(stock, start_date=today - datetime.timedelta(days=150),
                                              end_date=today - datetime.timedelta(days=1), frequency='60m').tail(150)
        # 持仓列表
        for stock in context.portfolio.positions.keys():
            # 且不在选股池列表中
            if stock not in context.stock_list:
                context.bar_60[stock] = get_price(stock, start_date=today - datetime.timedelta(days=150),
                                                  end_date=today - datetime.timedelta(days=1), frequency='60m').tail(
                    150)

        # -------------------------------------------#

        # 次新指数
        context.index_df = get_price('399678.XSHE', start_date=today - datetime.timedelta(days=300),
                                     end_date=today - datetime.timedelta(days=1), frequency='1d').tail(150)
        context.index_df = macd_alert_calculation(context.index_df)

        # 自选股评分
        stock_score(context)

        return None

    def handle_data(self, context, data):

        # 分钟线数据制作
        for stock in context.stock_list:
            self.handle_minute_data(context, data, stock)

        for stock in context.portfolio.positions.keys():
            if stock not in context.stock_list:
                self.handle_minute_data(context, data, stock)

        # 选股评分
        if context.timedelt % 60 == 0:
            stock_score(context)

    def handle_minute_data(self, context, data, stock):

        if context.timedelt % 60 == 0:
            temp_data = pd.DataFrame(
                {"low": history_bars(stock, 1, '60m', 'low')[0],
                 "open": "",
                 "high": history_bars(stock, 1, '60m', 'high')[0],
                 "volume": "",
                 "close": history_bars(stock, 1, '60m', 'close')[0],
                 "total_turnover": ""}, index=["0"])

            context.bar_60[stock] = context.bar_60[stock].append(temp_data, ignore_index=True)
            context.bar_60[stock] = macd_alert_calculation(context.bar_60[stock])

    def after_trading_end(self, context):

        stock_score(context)

        return None

    def __str__(self):
        return '分钟数据处理'


''' ----------------------统计类----------------------------'''


class Stat(Rule):
    def __init__(self, params):
        # 加载统计模块
        self.trade_total_count = 0
        self.trade_success_count = 0
        self.statis = {'win': [], 'loss': []}

    def after_trading_end(self, context):
        self.report(context)

    # todo: order 状态机制不同
    def when_sell_stock(self, position, order, is_normal):
        # if order.filled > 0:
        # 只要有成交，无论全部成交还是部分成交，则统计盈亏
        #   self.watch(position.security, order.filled, position.avg_cost, position.price)
        pass

    def reset(self):
        self.trade_total_count = 0
        self.trade_success_count = 0
        self.statis = {'win': [], 'loss': []}

    # 记录交易次数便于统计胜率
    # 卖出成功后针对卖出的量进行盈亏统计
    def watch(self, stock, sold_amount, avg_cost, cur_price):
        self.trade_total_count += 1
        current_value = sold_amount * cur_price
        cost = sold_amount * avg_cost

        percent = round((current_value - cost) / cost * 100, 2)
        if current_value > cost:
            self.trade_success_count += 1
            win = [stock, percent]
            self.statis['win'].append(win)
        else:
            loss = [stock, percent]
            self.statis['loss'].append(loss)

    def report(self, context):
        cash = context.portfolio.cash
        totol_value = context.portfolio.portfolio_value
        position = 1 - cash / totol_value
        # self.log_info("收盘后持仓概况:%s" % str(list(context.portfolio.positions)))
        # self.log_info("仓位概况:%.2f" % position)
        self.print_win_rate(context.now.strftime("%Y-%m-%d"), context.now.strftime("%Y-%m-%d"), context)

    # 打印胜率
    def print_win_rate(self, current_date, print_date, context):
        if str(current_date) == str(print_date):
            win_rate = 0
            if 0 < self.trade_total_count and 0 < self.trade_success_count:
                win_rate = round(self.trade_success_count / float(self.trade_total_count), 3)

            most_win = self.statis_most_win_percent()
            most_loss = self.statis_most_loss_percent()
            starting_cash = context.portfolio.starting_cash
            total_profit = self.statis_total_profit(context)
            if len(most_win) == 0 or len(most_loss) == 0:
                return

            s = '\n------------绩效报表------------'
            s += '\n交易次数: {0}, 盈利次数: {1}, 胜率: {2}'.format(self.trade_total_count, self.trade_success_count,
                                                          str(win_rate * 100) + str('%'))
            s += '\n单次盈利最高: {0}, 盈利比例: {1}%'.format(most_win['stock'], most_win['value'])
            s += '\n单次亏损最高: {0}, 亏损比例: {1}%'.format(most_loss['stock'], most_loss['value'])
            s += '\n总资产: {0}, 本金: {1}, 盈利: {2}, 盈亏比率：{3}%'.format(starting_cash + total_profit, starting_cash,
                                                                  total_profit, total_profit / starting_cash * 100)
            s += '\n--------------------------------'
            # self.log_info(s)

    # 统计单次盈利最高的股票
    def statis_most_win_percent(self):
        result = {}
        for statis in self.statis['win']:
            if {} == result:
                result['stock'] = statis[0]
                result['value'] = statis[1]
            else:
                if statis[1] > result['value']:
                    result['stock'] = statis[0]
                    result['value'] = statis[1]

        return result

    # 统计单次亏损最高的股票
    def statis_most_loss_percent(self):
        result = {}
        for statis in self.statis['loss']:
            if {} == result:
                result['stock'] = statis[0]
                result['value'] = statis[1]
            else:
                if statis[1] < result['value']:
                    result['stock'] = statis[0]
                    result['value'] = statis[1]

        return result

    # 统计总盈利金额
    def statis_total_profit(self, context):
        return context.portfolio.portfolio_value - context.portfolio.starting_cash

    def __str__(self):
        return '策略绩效统计'

        # 一下内容可另存为 util.py


# -*- coding:utf-8 -*-

# from kuanke.user_space_api import *

# import talib
# import pandas as pd
# import numpy as np

''' ==============================持仓操作函数，共用================================'''


# 开仓，买入指定价值的证券
# 报单成功并成交（包括全部成交或部分成交，此时成交量大于0），返回True
# 报单失败或者报单成功但被取消（此时成交量等于0），返回False
# 报单成功，触发所有规则的when_buy_stock函数

def open_position(security, value):
    order = order_target_value_(security, value)

    if order != None and order.filled_quantity > 0:
        for rule in g.all_rules:
            rule.when_buy_stock(security, order)
        return True
    return False


def open_position_by_percent(security, percent):
    order = order_target_value_by_percent(security, percent)

    if order != None and order.filled_quantity > 0:
        for rule in g.all_rules:
            rule.when_buy_stock(security, order)
        return True
    return False


# 平仓，卖出指定持仓
# 平仓成功并全部成交，返回True
# 报单失败或者报单成功但被取消（此时成交量等于0），或者报单非全部成交，返回False
# 报单成功，触发所有规则的when_sell_stock函数
def close_position(position, is_normal=True):
    security = position.order_book_id
    order = order_target_value_(security, 0)  # 可能会因停牌失败
    if order != None:
        # todo:
        # if order.filled > 0:
        for rule in g.all_rules:
            rule.when_sell_stock(position, order, is_normal)
        return True
    return False


# 平仓，卖出指定持仓
# 平仓成功并全部成交，返回True
# 报单失败或者报单成功但被取消（此时成交量等于0），或者报单非全部成交，返回False
# 报单成功，触发所有规则的when_sell_stock函数
def close_position_2(position, percentage, is_normal=True):
    # print('called')
    security = position.order_book_id
    # value = position.quantity * (1 - 0.3)
    # print(value)
    order = order_target_percent(security, percentage)  # 可能会因停牌失败
    if order != None:
        # todo:
        # if order.filled > 0:
        for rule in g.all_rules:
            rule.when_sell_stock(position, order, is_normal)
        return True
    return False


# 清空卖出所有持仓
# 清仓时，调用所有规则的 when_clear_position
def clear_position(context):
    if context.portfolio.positions:
        logger.info("==> 清仓，卖出所有股票")
        for stock in context.portfolio.positions.keys():
            position = context.portfolio.positions[stock]
            close_position(position, False)
    for rule in g.all_rules:
        rule.when_clear_position(context)

        # 自定义下单


# 根据Joinquant文档，当前报单函数都是阻塞执行，报单函数（如order_target_value）返回即表示报单完成
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


def order_target_value_by_percent(security, percent):
    if percent == 0:
        logger.debug("Selling out %s" % (security))
    else:
        logger.debug("Order %s to percent %f" % (security, percent))

    # 如果股票停牌，创建报单会失败，order_target_value 返回None
    # 如果股票涨跌停，创建报单会成功，order_target_value 返回Order，但是报单会取消
    # 部成部撤的报单，聚宽状态是已撤，此时成交量>0，可通过成交量判断是否有成交
    return order_target_percent(security, percent)


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~基础函数~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


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

    h = history_bars(stock, 4, '1d', ('close', 'open'))
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


# 计算MACD值
def calculate_macd(df):
    diff, dea, macd = talib.MACD(df['close'].values, 12, 26, 9)

    df['diff'] = pd.DataFrame(diff, index=df.index, columns=['diff'])
    df['dea'] = pd.DataFrame(dea, index=df.index, columns=['dea'])
    df['macd'] = pd.DataFrame(macd * 2, index=df.index, columns=['macd'])

    # print(df)

    return df


def macd_alert_calculation(macd_df):
    diff, dea, macd = talib.MACD(macd_df['close'].values, 12, 26, 9)

    # 计算MACD值
    macd_df['diff'] = pd.DataFrame(diff, index=macd_df.index, columns=['diff'])
    macd_df['dea'] = pd.DataFrame(dea, index=macd_df.index, columns=['dea'])
    macd_df['macd'] = pd.DataFrame(macd * 2, index=macd_df.index, columns=['macd'])

    # 1. 计算波峰跨度
    n1 = 0;
    n2 = 0;
    n3 = 0;

    return macd_df


def calculate_macd_index(df):
    diff, dea, macd = talib.MACD(df['close'].values, 12, 26, 9)

    df['diff'] = pd.DataFrame(diff, index=df.index, columns=['diff'])
    df['dea'] = pd.DataFrame(dea, index=df.index, columns=['dea'])
    df['macd'] = pd.DataFrame(macd * 2, index=df.index, columns=['macd'])

    return df


def stock_score(context):
    context.result_df = pd.DataFrame(columns={'stock', 'score', 'instruments', 'circulation_a', 'days_from_listed'})

    for stock in context.stock_list:

        score_df = context.bar_60[stock].tail(10)
        score_df = score_df.loc[:, ['close']]

        score_list = []

        for i in range(0, 10):
            h = context.bar_60[stock].tail(130 + i)
            h = h.iloc[0:130]

            low_price_130 = h.low.min()
            high_price_130 = h.high.max()

            avg_15_df = h.tail(15)
            avg_15 = avg_15_df.close.sum() / 15

            cur_price = h.iloc[-1]['close']

            score = ((cur_price - low_price_130) + (cur_price - high_price_130) + (cur_price - avg_15))
            score_list.append(score)

        score_list.reverse()
        score_df['score'] = pd.DataFrame(score_list, index=score_df.index, columns=['score'])

        temp_data = pd.DataFrame(
            {"score": score_df.iloc[-1]['score'],
             "stock": stock,
             "instruments": instruments(stock).symbol,
             "circulation_a": get_shares(stock, count=1, fields='circulation_a')[0] * context.bar_60[stock].iloc[-1][
                 'close'] / 10000,
             "days_from_listed": instruments(stock).days_from_listed()
             }, index=["0"])

        context.result_df = context.result_df.append(temp_data, ignore_index=True)
        context.score_df[stock] = score_df

    context.result_df = context.result_df.sort(columns='circulation_a', ascending=True)

    context.result_df['30_bottom_alert'] = pd.DataFrame(None, index=context.result_df.index,
                                                        columns=['30_bottom_alert'])

    print(context.result_df)

    context.stock_list = list(context.result_df.stock)

    return None


# 获取股票n日以来涨幅，根据当前价计算
# n 默认20日
def get_growth_rate(security, n=20):
    lc = get_close_price(security, n)
    c = get_close_price(security, 1)

    if not isnan(lc) and not isnan(c) and lc != 0:
        return (c - lc) / lc
    else:
        log.error("数据非法, security: %s, %d日收盘价: %f, 当前价: %f" % (security, n, lc, c))
        return 0


# 获取前n个单位时间当时的收盘价
def get_close_price(security, n, unit='1d'):
    return history_bars(security, n, unit, ('close'))[0]


def isnan(value):
    if value == None:
        return True
    else:
        return False


def findATR(context, bar, stock):
    history_df = bar[stock].tail(context.ATRperiod + 2)

    close = history_df['close'][0:context.ATRperiod]
    high = history_df['high'][1:context.ATRperiod + 1]
    low = history_df['low'][1:context.ATRperiod + 1]

    art1 = high.values - low.values
    art2 = abs(close.values - high.values)
    art3 = abs(close.values - low.values)
    art123 = np.matrix([art1, art2, art3])

    rawatr = np.array(art123.max(0)).flatten()
    ATR = rawatr.sum() / len(rawatr)

    return ATR


def createdic(context, data, stock):
    if stock not in context.maxvalue.columns:
        temp = pd.DataFrame({str(stock): [0]})
        context.maxvalue = pd.concat([context.maxvalue, temp], axis=1, join='inner')

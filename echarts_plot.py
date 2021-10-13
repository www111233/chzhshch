# coding: utf-8
"""
# https://github.com/zengbin93/czsc
使用 pyecharts 定制绘图模块

"""

from pyecharts import options as opts
from pyecharts.charts import HeatMap, Kline, Line, Bar, Scatter, Grid, Boxplot
from pyecharts.commons.utils import JsCode
from typing import List
import numpy as np

from pyecharts import types
from pyecharts.charts.chart import RectChart
from pyecharts.globals import ChartType

def SMA(close: np.array, timeperiod=5):
    """简单移动平均

    https://baike.baidu.com/item/%E7%A7%BB%E5%8A%A8%E5%B9%B3%E5%9D%87%E7%BA%BF/217887

    :param close: np.array
        收盘价序列
    :param timeperiod: int
        均线参数
    :return: np.array
    """
    res = []
    for i in range(len(close)):
        if i < timeperiod:
            seq = close[0: i+1]
        else:
            seq = close[i - timeperiod + 1: i + 1]
        res.append(seq.mean())
    return np.array(res, dtype=np.double)

def EMA(close: np.array, timeperiod=5):
    """
    https://baike.baidu.com/item/EMA/12646151

    :param close: np.array
        收盘价序列
    :param timeperiod: int
        均线参数
    :return: np.array
    """
    res = []
    for i in range(len(close)):
        if i < 1:
            res.append(close[i])
        else:
            ema = (2 * close[i] + res[i-1] * (timeperiod-1)) / (timeperiod+1)
            res.append(ema)
    return np.array(res, dtype=np.double)

def MACD(close: np.array, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD 异同移动平均线
    https://baike.baidu.com/item/MACD%E6%8C%87%E6%A0%87/6271283

    :param close: np.array
        收盘价序列
    :param fastperiod: int
        快周期，默认值 12
    :param slowperiod: int
        慢周期，默认值 26
    :param signalperiod: int
        信号周期，默认值 9
    :return: (np.array, np.array, np.array)
        diff, dea, macd
    """
    ema12 = EMA(close, timeperiod=fastperiod)
    ema26 = EMA(close, timeperiod=slowperiod)
    diff = ema12 - ema26
    dea = EMA(diff, timeperiod=signalperiod)
    macd = (diff - dea) * 2
    return diff, dea, macd

def KDJ(close: np.array, high: np.array, low: np.array):
    """

    :param close: 收盘价序列
    :param high: 最高价序列
    :param low: 最低价序列
    :return:
    """
    n = 9
    hv = []
    lv = []
    for i in range(len(close)):
        if i < n:
            h_ = high[0: i+1]
            l_ = low[0: i+1]
        else:
            h_ = high[i - n + 1: i + 1]
            l_ = low[i - n + 1: i + 1]
        hv.append(max(h_))
        lv.append(min(l_))

    hv = np.around(hv, decimals=2)
    lv = np.around(lv, decimals=2)
    rsv = np.where(hv == lv, 0, (close - lv) / (hv - lv) * 100)

    k = []
    d = []
    j = []
    for i in range(len(rsv)):
        if i < n:
            k_ = rsv[i]
            d_ = k_
        else:
            k_ = (2 / 3) * k[i-1] + (1 / 3) * rsv[i]
            d_ = (2 / 3) * d[i-1] + (1 / 3) * k_

        k.append(k_)
        d.append(d_)
        j.append(3 * k_ - 2 * d_)

    k = np.array(k, dtype=np.double)
    d = np.array(d, dtype=np.double)
    j = np.array(j, dtype=np.double)
    return k, d, j

def RSQ(close: [np.array, list]) -> float:
    """拟合优度 R SQuare

    :param close: 收盘价序列
    :return:
    """
    x = list(range(len(close)))
    y = np.array(close)
    x_squred_sum = sum([x1 * x1 for x1 in x])
    xy_product_sum = sum([x[i] * y[i] for i in range(len(x))])
    num = len(x)
    x_sum = sum(x)
    y_sum = sum(y)
    delta = float(num * x_squred_sum - x_sum * x_sum)
    if delta == 0:
        return 0
    y_intercept = (1 / delta) * (x_squred_sum * y_sum - x_sum * xy_product_sum)
    slope = (1 / delta) * (num * xy_product_sum - x_sum * y_sum)

    y_mean = np.mean(y)
    ss_tot = sum([(y1 - y_mean) * (y1 - y_mean) for y1 in y]) + 0.00001
    ss_err = sum([(y[i] - slope * x[i] - y_intercept) * (y[i] - slope * x[i] - y_intercept) for i in range(len(x))])
    rsq = 1 - ss_err / ss_tot

    return round(rsq, 4)


def heat_map(data: List[dict],
             x_label: List[str] = None,
             y_label: List[str] = None,
             title: str = "热力图",
             width: str = "900px",
             height: str = "680px") -> HeatMap:
    """绘制热力图

    :param data: 用于绘制热力图的数据，示例如下
        [{'x': '0hour', 'y': '0day', 'heat': 11},
         {'x': '0hour', 'y': '1day', 'heat': 40},
         {'x': '0hour', 'y': '2day', 'heat': 38},
         {'x': '0hour', 'y': '3day', 'heat': 36},
         {'x': '0hour', 'y': '4day', 'heat': 11}]
    :param x_label: x轴标签
    :param y_label: y轴标签
    :param title: 图表标题
    :param width: 图表宽度
    :param height: 图表高度
    :return: 图表
    """

    value = [[s['x'], s['y'], s['heat']] for s in data]
    heat = [s['heat'] for s in data]

    if not x_label:
        x_label = sorted(list(set([s['x'] for s in data])))

    if not y_label:
        y_label = sorted(list(set([s['y'] for s in data])))

    vis_map_opts = opts.VisualMapOpts(pos_left="90%", pos_top="20%", min_=min(heat), max_=max(heat))
    title_opts = opts.TitleOpts(title=title)
    init_opts = opts.InitOpts(page_title=title, width=width, height=height)
    dz_inside = opts.DataZoomOpts(False, "inside", xaxis_index=[0], range_start=80, range_end=100)
    dz_slider = opts.DataZoomOpts(True, "slider", xaxis_index=[0], pos_top="96%", pos_bottom="0%",
                                  range_start=80, range_end=100)
    legend_opts = opts.LegendOpts(is_show=False)

    hm = HeatMap(init_opts=init_opts)
    hm.add_xaxis(x_label)
    hm.add_yaxis("heat", y_label, value, label_opts=opts.LabelOpts(is_show=True, position="inside"))
    hm.set_global_opts(title_opts=title_opts, visualmap_opts=vis_map_opts, legend_opts=legend_opts,
                       xaxis_opts=opts.AxisOpts(grid_index=0), datazoom_opts=[dz_inside, dz_slider])
    return hm


def kline_pro(kline: List[dict],
              fx: List[dict] = None,
              bi: List[dict] = None,
              xd: List[dict] = None,
              xd2: List[dict] = None,
              bs: List[dict] = None,
              mark:List[dict] = None,
              title: str = "缠中说禅K线分析",
              width: str = "880px",
              height: str = '1080px') -> Grid:
    """绘制缠中说禅K线分析结果

    :param kline: K线
    :param fx: 分型识别结果
    :param bi: 笔识别结果
        {'dt': Timestamp('2020-11-26 00:00:00'),
          'fx_mark': 'd',
          'start_dt': Timestamp('2020-11-25 00:00:00'),
          'end_dt': Timestamp('2020-11-27 00:00:00'),
          'fx_high': 144.87,
          'fx_low': 138.0,
          'bi': 138.0}
    :param xd: 线段识别结果
    :param bs: 买卖点
    :param title: 图表标题
    :param width: 图表宽度
    :param height: 图表高度
    :return: 用Grid组合好的图表
    """
    # 配置项设置
    # ------------------------------------------------------------------------------------------------------------------
    bg_color = "#1f212d"  # 背景
    up_color = "#F9293E"
    down_color = "#00aa3b"
    mark_opts = opts.MarkPointOpts(mark)

    init_opts = opts.InitOpts(bg_color=bg_color, width=width, height=height, animation_opts=opts.AnimationOpts(False))
    title_opts = opts.TitleOpts(title=title, pos_top="1%",
                                title_textstyle_opts=opts.TextStyleOpts(color=up_color, font_size=20),
                                subtitle_textstyle_opts=opts.TextStyleOpts(color=down_color, font_size=12))

    label_not_show_opts = opts.LabelOpts(is_show=False)
    legend_not_show_opts = opts.LegendOpts(is_show=False)
    red_item_style = opts.ItemStyleOpts(color=up_color)
    green_item_style = opts.ItemStyleOpts(color=down_color)
    k_style_opts = opts.ItemStyleOpts(color=up_color, color0=down_color, border_color=up_color,
                                      border_color0=down_color, opacity=0.8)

    legend_opts = opts.LegendOpts(is_show=True, pos_top="1%", pos_left="30%", item_width=14, item_height=8,
                                  textstyle_opts=opts.TextStyleOpts(font_size=12, color="#0e99e2"))
    brush_opts = opts.BrushOpts(tool_box=["rect", "polygon", "keep", "lineX", "lineY", "clear"],
                                x_axis_index="all", brush_link="all",
                                out_of_brush={"colorAlpha": 0.1}, brush_type="lineX")

    axis_pointer_opts = opts.AxisPointerOpts(is_show=True, link=[{"xAxisIndex": "all"}])

    dz_inside = opts.DataZoomOpts(False, "inside", xaxis_index=[0, 1, 2], range_start=80, range_end=100)
    dz_slider = opts.DataZoomOpts(True, "slider", xaxis_index=[0, 1, 2], pos_top="96%",
                                  pos_bottom="0%", range_start=80, range_end=100)

    yaxis_opts = opts.AxisOpts(is_scale=True,
                               axislabel_opts=opts.LabelOpts(color="#c7c7c7", font_size=8, position="inside"))

    grid0_xaxis_opts = opts.AxisOpts(type_="category", grid_index=0, axislabel_opts=label_not_show_opts,
                                     split_number=20, min_="dataMin", max_="dataMax",
                                     is_scale=True, boundary_gap=False,
                                     axisline_opts=opts.AxisLineOpts(is_on_zero=False))

    tool_tip_opts = opts.TooltipOpts(
        trigger="axis",
        axis_pointer_type="cross",
        background_color="rgba(245, 245, 245, 0.8)",
        border_width=1,
        border_color="#ccc",
        position=JsCode("""
                    function (pos, params, el, elRect, size) {
    					var obj = {top: 10};
    					obj[['left', 'right'][+(pos[0] < size.viewSize[0] / 2)]] = 30;
    					return obj;
    				}
                    """),
        textstyle_opts=opts.TextStyleOpts(color="#000"),
    )

    # 数据预处理
    # ------------------------------------------------------------------------------------------------------------------
    dts = [x['dt'] for x in kline]
    # k_data = [[x['open'], x['close'], x['low'], x['high']] for x in kline]
    k_data = [opts.CandleStickItem(name=i, value=[x['open'], x['close'], x['low'], x['high']])
              for i, x in enumerate(kline)]

    vol = []
    for i, row in enumerate(kline):
        item_style = red_item_style if row['close'] > row['open'] else green_item_style
        bar = opts.BarItem(name=i, value=row['vol'], itemstyle_opts=item_style, label_opts=label_not_show_opts)
        vol.append(bar)

    close = np.array([x['close'] for x in kline], dtype=np.double)
    diff, dea, macd = MACD(close)

    ma5 = SMA(close, timeperiod=5)
    ma34 = SMA(close, timeperiod=34)
    ma233 = SMA(close, timeperiod=233)

    macd_bar = []
    for i, v in enumerate(macd.tolist()):
        item_style = red_item_style if v > 0 else green_item_style
        bar = opts.BarItem(name=i, value=round(v, 4), itemstyle_opts=item_style,
                           label_opts=label_not_show_opts)
        macd_bar.append(bar)

    diff = diff.round(4)
    dea = dea.round(4)

    # K 线主图
    # ------------------------------------------------------------------------------------------------------------------
    chart_k = Kline()
    chart_k.add_xaxis(xaxis_data=dts)
    chart_k.add_yaxis(series_name="Kline", markpoint_opts=mark_opts, y_axis=k_data, itemstyle_opts=k_style_opts)

    chart_k.set_global_opts(
        legend_opts=legend_opts,
        datazoom_opts=[dz_inside, dz_slider],
        yaxis_opts=yaxis_opts,
        tooltip_opts=tool_tip_opts,
        axispointer_opts=axis_pointer_opts,
        brush_opts=brush_opts,
        title_opts=title_opts,
        xaxis_opts=grid0_xaxis_opts
    )

    # 均线图
    # ------------------------------------------------------------------------------------------------------------------
    chart_ma = Line()
    chart_ma.add_xaxis(xaxis_data=dts)

    ma_keys = {"MA5": ma5, "MA34": ma34, "MA233": ma233}
    ma_colors = ["#39afe6", "#da6ee8", "#00940b"]
    for i, (name, ma) in enumerate(ma_keys.items()):
        chart_ma.add_yaxis(series_name=name, y_axis=ma, is_smooth=True,
                           is_selected=False, symbol_size=0, label_opts=label_not_show_opts,
                           linestyle_opts=opts.LineStyleOpts(opacity=0.8, width=1.0, color=ma_colors[i]))

    chart_ma.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
    chart_k = chart_k.overlap(chart_ma)

    # 缠论结果
    # ------------------------------------------------------------------------------------------------------------------
    if fx:
        fx_dts = [x['dt'] for x in fx]
        fx_val = [x['fx'] for x in fx]
        chart_fx = Scatter()
        chart_fx.add_xaxis(fx_dts)
        chart_fx.add_yaxis(series_name="FX", y_axis=fx_val, is_selected=False,
                           symbol="circle", symbol_size=6, label_opts=label_not_show_opts,
                           itemstyle_opts=opts.ItemStyleOpts(color="rgba(152, 147, 193, 1.0)", ))

        chart_fx.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
        chart_k = chart_k.overlap(chart_fx)

    if bi:
        bi_dts = [x['dt'] for x in bi]
        bi_val = [x['bi'] for x in bi]
        chart_bi = Line()
        chart_bi.add_xaxis(bi_dts)
        chart_bi.add_yaxis(series_name="BI", y_axis=bi_val, is_selected=True,
                           symbol="diamond", symbol_size=10, label_opts=label_not_show_opts,
                           itemstyle_opts=opts.ItemStyleOpts(color="rgba(255, 255, 0, 1.0)", ),
                           linestyle_opts=opts.LineStyleOpts(width=1.5))

        chart_bi.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
        chart_k = chart_k.overlap(chart_bi)

    if xd:
        xd_dts = [x['dt'] for x in xd]
        xd_val = [x['xd'] for x in xd]
        chart_xd = Line()
        chart_xd.add_xaxis(xd_dts)
        chart_xd.add_yaxis(series_name="XD", y_axis=xd_val, is_selected=True, symbol="triangle", symbol_size=10,
                           itemstyle_opts=opts.ItemStyleOpts(color="rgba(37, 141, 54, 1.0)", ),
                           linestyle_opts=opts.LineStyleOpts(width=3.5))
        chart_xd.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
        chart_k = chart_k.overlap(chart_xd)

    if xd2:
        xd_dts2 = [x['dt'] for x in xd2]
        xd_val2 = [x['xd'] for x in xd2]
        chart_xd2 = Line()
        chart_xd2.add_xaxis(xd_dts2)
        chart_xd2.add_yaxis(series_name="XD2", y_axis=xd_val2, is_selected=True, symbol="triangle", symbol_size=10,
                           itemstyle_opts=opts.ItemStyleOpts(color="rgba(0xe2, 0x3a, 0x92, 1.0)", ),
                           linestyle_opts=opts.LineStyleOpts(width=4.5))
        chart_xd.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
        chart_k = chart_k.overlap(chart_xd2)

    if bs:
        b_dts = [x['dt'] for x in bs if x['mark'] == 'buy']
        if len(b_dts) > 0:
            b_val = [x['price'] for x in bs if x['mark'] == 'buy']
            chart_b = Scatter()
            chart_b.add_xaxis(b_dts)
            chart_b.add_yaxis(series_name="BUY", y_axis=b_val, is_selected=False, symbol="arrow", symbol_size=8,
                              itemstyle_opts=opts.ItemStyleOpts(color="#f31e1e", ))

            chart_b.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
            chart_k = chart_k.overlap(chart_b)

        s_dts = [x['dt'] for x in bs if x['mark'] == 'sell']
        if len(s_dts) > 0:
            s_val = [x['price'] for x in bs if x['mark'] == 'sell']
            chart_s = Scatter()
            chart_s.add_xaxis(s_dts)
            chart_s.add_yaxis(series_name="SELL", y_axis=s_val, is_selected=False, symbol="pin", symbol_size=12,
                              itemstyle_opts=opts.ItemStyleOpts(color="#45b97d", ))

            chart_s.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
            chart_k = chart_k.overlap(chart_s)

    # 成交量图
    # ------------------------------------------------------------------------------------------------------------------
    chart_vol = Bar()
    chart_vol.add_xaxis(dts)
    chart_vol.add_yaxis(series_name="Volume", y_axis=vol, bar_width='60%')
    chart_vol.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="category",
            grid_index=1,
            axislabel_opts=opts.LabelOpts(is_show=True, font_size=8, color="#9b9da9"),
        ),
        yaxis_opts=yaxis_opts, legend_opts=legend_not_show_opts,
    )

    # MACD图
    # ------------------------------------------------------------------------------------------------------------------
    chart_macd = Bar()
    chart_macd.add_xaxis(dts)
    chart_macd.add_yaxis(series_name="MACD", y_axis=macd_bar, bar_width='60%')
    chart_macd.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="category",
            grid_index=2,
            axislabel_opts=opts.LabelOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            grid_index=2,
            split_number=4,
            axisline_opts=opts.AxisLineOpts(is_on_zero=False),
            axistick_opts=opts.AxisTickOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
            axislabel_opts=opts.LabelOpts(is_show=True, color="#c7c7c7"),
        ),
        legend_opts=opts.LegendOpts(is_show=False),
    )

    line = Line()
    line.add_xaxis(dts)
    line.add_yaxis(series_name="DIFF", y_axis=diff, label_opts=label_not_show_opts, is_symbol_show=False,
                   linestyle_opts=opts.LineStyleOpts(opacity=0.8, width=1.0, color="#da6ee8"))
    line.add_yaxis(series_name="DEA", y_axis=dea, label_opts=label_not_show_opts, is_symbol_show=False,
                   linestyle_opts=opts.LineStyleOpts(opacity=0.8, width=1.0, color="#39afe6"))

    chart_macd = chart_macd.overlap(line)

    grid0_opts = opts.GridOpts(pos_left="0%", pos_right="1%", pos_top="12%", height="58%")
    grid1_opts = opts.GridOpts(pos_left="0%", pos_right="1%", pos_top="74%", height="8%")
    grid2_opts = opts.GridOpts(pos_left="0%", pos_right="1%", pos_top="86%", height="10%")

    grid_chart = Grid(init_opts)
    grid_chart.add(chart_k, grid_opts=grid0_opts)
    grid_chart.add(chart_vol, grid_opts=grid1_opts)
    grid_chart.add(chart_macd, grid_opts=grid2_opts)
    return grid_chart


def box_plot(data: dict,
             title: str = "箱线图",
             width: str = "900px",
             height: str = "680px") -> Boxplot:
    """

    :param data: 数据
        样例：
        data = {
            "expr 0": [960, 850, 830, 880],
            "expr 1": [960, 850, 830, 880],
        }
    :param title:
    :param width:
    :param height:
    :return:
    """
    x_data = []
    y_data = []
    for k, v in data.items():
        x_data.append(k)
        y_data.append(v)

    init_opts = opts.InitOpts(page_title=title, width=width, height=height)

    chart = Boxplot(init_opts=init_opts)
    chart.add_xaxis(xaxis_data=x_data)
    chart.add_yaxis(series_name="", y_axis=y_data)
    chart.set_global_opts(title_opts=opts.TitleOpts(pos_left="center", title=title),
                          tooltip_opts=opts.TooltipOpts(trigger="item", axis_pointer_type="shadow"),
                          xaxis_opts=opts.AxisOpts(
                              type_="category",
                              boundary_gap=True,
                              splitarea_opts=opts.SplitAreaOpts(is_show=False),
                              axislabel_opts=opts.LabelOpts(formatter="{value}"),
                              splitline_opts=opts.SplitLineOpts(is_show=False),
                          ),
                          yaxis_opts=opts.AxisOpts(
                              type_="value",
                              name="",
                              splitarea_opts=opts.SplitAreaOpts(
                                  is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                              )
                          ))
    return chart

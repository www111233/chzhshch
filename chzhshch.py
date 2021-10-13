# coding: utf-8
# cython: language_level=3
import pdb
from enum import Enum
from importlib import reload
import re, struct, time, sys, builtins, math, traceback
from datetime import datetime, timedelta
from typing import List, Union
import logging
import logging.handlers
import datetime as _datetime

try:
    from termcolor import colored
except:
    def colored(text, color=None, on_color=None, attrs=None):
        return text
import requests
import inspect
import typing
from contextlib import suppress
from functools import wraps
def enforce_types(callable):
    spec = inspect.getfullargspec(callable)

    def check_types(*args, **kwargs):
        parameters = dict(zip(spec.args, args))
        parameters.update(kwargs)
        for name, value in parameters.items():
            with suppress(KeyError): # Assume un-annotated parameters can be any type
                type_hint = spec.annotations[name]
                if isinstance(type_hint, typing._SpecialForm):
                    # No check for typing.Any, typing.Union, typing.ClassVar (without parameters)
                    continue

                try:
                    actual_type = type_hint.__origin__
                except AttributeError:
                    # In case of non-typing types (such as , for instance)
                    actual_type = type_hint
                # In Python 3.8 one would replace the try/except with
                # actual_type = typing.get_origin(type_hint) or type_hint

                if isinstance(actual_type, typing._SpecialForm):
                    # case of typing.Union[…] or typing.ClassVar[…]
                    actual_type = type_hint.__args__
                if isinstance(actual_type, list):
                    actual_type = tuple(actual_type)
                if not isinstance(value, actual_type):
                    raise TypeError('Unexpected type for \'{}\' (expected {} but found {})'.format(name, type_hint, type(value)))

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            check_types(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper

    if inspect.isclass(callable):
        callable.__init__ = decorate(callable.__init__)
        return callable

    return decorate(callable)

class Config:
    def __init__(self):
        self.__debug = 0
        logger = logging.getLogger('chzhshch')
        logger.setLevel(logging.DEBUG)
        
        rf_handler = logging.handlers.TimedRotatingFileHandler('all.log', when='midnight', interval=1, backupCount=7, atTime=_datetime.time(0, 0, 0, 0))
        rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        f_handler = logging.FileHandler('chzhshch-error.log')
        f_handler.setLevel(logging.WARN)
        f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
        
        logger.addHandler(rf_handler)
        logger.addHandler(f_handler)
        self.logger = logger
    
    @property
    def debug(self):
        return self.__debug
        
    @debug.setter
    def debug(self, n):
        self.__debug = n
        if n:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        

config = Config()
config. debug = 0





class Logging(object):
    def __init__(self, name, level=0):
        self.level = level
        self.outstream = sys.stderr
        self.name = name

    def __print(self, *args, **kwords):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        args = list(args)
        for i in range(len(args)):
            string = args[i]
            if type(string) is str and '\x1b[0m' in string:
                regex = re.compile('(\\x1b\[[\d]+m)')
                find = regex.findall(args[i])
                for f in find:
                    string = string.replace(f, '')
            args[i] = string
        builtins.print(now, *args, **kwords)
        self.outstream.flush()

    def debug(self, *args):
        if self.level >= 0:
            self.__print(self.level, *args, file=self.outstream)

    def info(self, *args):
        if self.level >= 1:
            self.__print(self.level, *args, file=self.outstream)

def print(*args, **kwords):
    result = []
    for i in range(len(args)):
        if args[i] == Shape.G:
            result.append(colored(args[i], "red"))
        elif args[i] == Shape.D:
            result.append(colored(args[i], "green"))
        elif str(args[i]) == "True":
            result.append(colored(args[i], "red"))
        elif str(args[i]) == "False":
            result.append(colored(args[i], "green"))
        elif args[i] == "少阳":
            result.append(colored(args[i], "green"))
        elif args[i] == "老阳":
            result.append(colored(args[i], "red"))
        elif args[i] == "少阴":
            result.append(colored(args[i], "yellow"))
        elif args[i] == "老阴":
            result.append(colored(args[i], "blue"))
        else:
            result.append(args[i])
    builtins.print(*tuple(result), **kwords)

def timeit(func):
    def run(*args, **kwords):
        t = time.perf_counter()
        result = func(*args, **kwords)
        e = time.perf_counter() - t
        print(f'func {func.__name__} coast time:{e:.8f} s')
        return result
    return run

class Machine(object):
    __slots__ = ["__status", "__states", "__state"]
    def __init__(self, status=[0, ]):
        self.__status = status
        self.__state = 0
        self.__states = [self.__state] # 状态缓存

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, state):
        if state in self.__status:
            self.__state = state
            self.__states.append(state)
            if len(self.__states) > 8:
                self.__states = self.__states[-8:]
        else:
            raise ValueError(state, "not in status")

    def doBack(self):
        '''退回到上一状态'''
        if len(self.__states) < 2:return False
        self.__states.pop() # 当前状态
        self.__state = self.__states.pop()
        return True

class FeatureMachine:
    def __init__(self):
        self._left, self._mid, self._right = (None, None, None)
        self._deal_left, self._deal_mid, self._deal_right = (1,0,0)

    def setMid(self, t, n):
        #print(n, t)
        self._mid = t

    def setLeft(self, t, n):
        #print(n, t)
        self._left = t

    def setRight(self, t, n):
        #print(n, t)
        self._right = t

class Stack(object):
    def __init__(self):
        self.stack = []

    def push(self, data):
        """
        进栈函数
        """
        self.stack.append(data)

    def pop(self):
        """
        出栈函数，
        """
        return self.stack.pop()

    def gettop(self):
        """
        取栈顶
        """
        return self.stack[-1]

class ChanException(Exception):
    pass

class Freq(Enum):
    Tick = "Tick"
    F1 = "1分钟"
    F5 = "5分钟"
    F15 = "15分钟"
    F30 = "30分钟"
    F60 = "60分钟"
    D = "日线"
    W = "周线"
    M = "月线"
    S = "季线"
    Y = "年线"

class Shape(Enum):
    # 三根k线的所有分型
    D = "底分型"
    G = "顶分型"
    S = "上升分型"
    X = "下降分型"
    #T = "喇叭口型"

class Direction(Enum):
    # 两根k线的所有关系
    Up = "上涨"
    Down = "下跌"
    JumpUp = "跳涨"
    JumpDown = "跳跌"

    Left = "左包右" # 顺序包含
    Right = "右包左" # 逆序包含

    Unknow = "未知"

@enforce_types
class Signal(object):
    def __init__(self, name: str, level: int):
        self.name = name
        self.level = level

@enforce_types
class RawCandle(object):
    """原始K线"""
    #__slots__ = ["symbol", "dt", "open", "high", "low", "close", "vol", "style", "index", "direction"]
    def __init__(self, symbol: str, dt: datetime, open: [float, int], high: [float, int],
        low: [float, int], close: [float, int], vol: [float, int],
        freq: Freq = None, style: Shape = None, index: int = 0):

        self.symbol = symbol
        self.dt = dt
        # freq: str = None
        self.open = float(open)
        self.high = float(high)
        self.low = float(low)
        self.close = float(close)
        self.vol = float(vol)
        self.freq = freq
        self.style = style
        self.index = index
        self.direction = Direction.Down if self.open > self.close else Direction.Up

        self.ma5 = None
        self.ma10 = None
        self.ma20 = None
        self.ema12 = None
        self.ema26 = None
        self.diff = None
        self.dea = None
        self.macd = None

        self.K = 50
        self.D = 50
        self.J = 50
        self.mark = dict()

    @classmethod
    def frombytes(cls, buf:bytes, symbol:str):
        timestamp, open, high, low, close, vol = struct.unpack(">6d", buf)
        return cls(symbol, datetime.fromtimestamp(timestamp), open, high, low, close, vol)
    def __bytes__(self):
        return struct.pack(">6d", int(self.dt.timestamp()), round(self.open, 8), round(self.high, 8), round(self.low, 8), round(self.close, 8), round(self.vol, 8))

    def __str__(self):
        return f'RawCandle("{self.symbol}", {self.dt}, {self.open}, {self.high}, {self.low}, {self.close}, {self.vol}, {self.style})'
    def __repr__(self):
        return f'RawCandle("{self.symbol}", {self.dt}, {self.open}, {self.high}, {self.low}, {self.close}, {self.vol}, {self.style})'
    def __eq__(self, other):
        return (isinstance(other, type(self))) and \
               ((self.symbol, self.dt, self.open, self.high, self.low, self.close, self.vol, self.freq) == (other.symbol, other.dt, other.open, other.high, other.low, other.close, other.vol, other.freq))

    def __hash__(self):
        return (hash(self.symbol) ^ hash(self.dt) ^ hash(self.open) ^ hash(self.high) ^ hash(self.low) ^ hash(self.close) ^ hash(self.vol) ^ hash(self.freq) ^ hash((self.symbol, self.dt, self.open, self.high, self.low, self.close, self.vol, self.freq)))
    def candleDict(self):
        return {"symbol":self.symbol, "dt": self.dt, "open":self.open,
                         "high":self.high, "low":self.low, "close":self.close,
                         "vol":self.vol, "style":self.style, "index":self.index}
    def toChanCandle(self):
        return ChanCandle(self.symbol, self.high if self.direction is Direction.Down else self.low, self.low if self.direction is Direction.Down else self.high, [self, ])
    def transform(self, obj):
        return obj(symbol=self.symbol, dt=self.dt, open=self.open, high=self.high, low=self.low, close=self.close, vol=self.vol)

def binCandle(arr:List[RawCandle], freq:Freq):
    symbol = arr[0].symbol
    dt = arr[0].dt
    open = arr[0].open
    high = max([k.high for k in arr])
    low = min([k.low for k in arr])
    close = arr[-1].close
    vol = sum([k.vol for k in arr])
    return RawCandle(symbol, dt, open, high, low, close, vol, freq)

@enforce_types
class ChanCandle:
    """缠论k线"""
    __slots__ = ["symbol", "start", "end", "style", "index", "elements", "info"]
    def __init__(self, symbol: str, open: [float, int] = None, close: [float, int] = None, elements: List[RawCandle] = None, level=-1):
        self.symbol = symbol
        self.start = open
        self.end = close
        self.elements = elements
        self.style = None
        self.index = 0
        self.info = ""

    def __getitem__(self, i):
        return self.elements[i]
    def __str__(self):
        return f'ChanCandle("{self.symbol}", {self.dt}, {self.start}, {self.end}, {self.style}, {self.direction}, size={len(self.elements)})'
    def __repr__(self):
        return f'ChanCandle("{self.symbol}", {self.dt}, {self.start}, {self.end}, {self.style}, {self.direction}, size={len(self.elements)})'

    @property
    def dt(self) -> datetime:
        if not self.elements:
            return datetime.now()
        return self.elements[-1].dt

    @property
    def vol(self) -> [int, float]:
        if not self.elements:
            return 0
        return sum([i.vol for i in self.elements])

    @property
    def realdt(self) -> datetime:
        if not self.elements:
            return None
        if self.style is Shape.G:
            return max(self.elements, key=lambda x:x.high).dt
        elif self.style is Shape.D:
            return min(self.elements, key=lambda x:x.low).dt
        else:
            return None
            raise ChanException

    @property
    def macd(self) -> [float, int]:
        if not self.elements:
            return 0
        return sum([abs(o.macd) for o in self.elements])

    @property
    def high(self) -> [float, int]:
        return max(self.start, self.end)
    @property
    def low(self) -> [float, int]:
        return min(self.start, self.end)

    @property
    def direction(self) -> Direction:
        if self.start > self.end:return Direction.Down
        elif self.start < self.end:return Direction.Up
        else:return Direction.Unknow

    def candleDict(self) -> dict:
        return {"symbol":self.symbol, "dt": self.dt, "open":self.start,
                         "high":self.high, "low":self.low, "close":self.end,
                         "vol":self.vol, "style":self.style, "index":self.index, "dir":self.direction, "elements":self.elements}

@enforce_types
class ChanFenXing: # 分型
    __slots__ = ["style", "fx", "elements"]
    def __init__(self, left:ChanCandle, mid:ChanCandle, right:ChanCandle):
        self.style = tripleRelation(left, mid, right, False)
        if self.style == None:raise ChanException("無法解析分型")
        if not (left.dt < mid.dt < right.dt):
            raise ChanException("時序錯誤")
        self.elements = (left, mid, right)
        mid.style = self.style
        self.fx = None
        if self.style is Shape.G:
            self.fx = self.high

        if self.style is Shape.D:
            self.fx = self.low

    def __eq__(self, other):
        return (isinstance(other, type(self))) and \
               ((self.l, self.m, self.r) == (other.l, other.m, other.r))

    def __str__(self):
        return f'ChanFenXing("{self.style}", {self.dt}, {self.fx})'

    def __repr__(self):
        return f'ChanFenXing("{self.style}", {self.dt}, {self.fx})'

    @property
    def dt(self) -> datetime:
        return self.m.dt
    @property
    def symbol(self) -> str:
        return self.m.symbol
    @property
    def power(self) -> str:
        if self.style is Shape.G:
            return "强" if self.r.low < self.l.low else "弱"

        if self.style is Shape.D:
            return "强" if self.r.high > self.l.high else "弱"

    @property
    def low(self) -> [float, int]:
        return self.m.low
    @property
    def high(self) -> [float, int]:
        return self.m.high

    @property
    def l(self) -> ChanCandle:
        return self.elements[0]
    @property
    def m(self) -> ChanCandle:
        return self.elements[1]
    @property
    def r(self) -> ChanCandle:
        return self.elements[2]

@enforce_types
class ChanFeature:
    __slots__ = ["__level", "__symbol", "__start", "__end", "elements", "index", "isFixed", "isVisual"]
    def __init__(self, start:ChanFenXing, end:ChanFenXing, elements:List[ChanCandle], level:int=0, index:int=0, isVisual:bool=False, isFixed:int=1):
        if start.symbol != end.symbol:raise ChanException
        self.__symbol = start.symbol

        if not (start.dt < end.dt):
            raise ValueError("时序错误")

        if (start.style == Shape.D) and (end.style == Shape.G):
            if start.low > end.high:
                raise ValueError("DG 结构错误: 上涨特征序列 起点比终点高", level, start.dt, end.dt)
        elif (start.style == Shape.G) and (end.style == Shape.D):
            if start.high < end.low:
                raise ValueError("GD 结构错误: 下跌特征序列 起点比终点低", level, start.dt, end.dt)

        elif (start.style == Shape.D) and (end.style == Shape.S):
            if start.low > end.r.high:
                raise ValueError("DS 结构错误: 上涨特征序列 起点比终点高", level, start.dt, end.dt)
        elif (start.style == Shape.G) and (end.style == Shape.X):
            if start.high < end.r.low:
                raise ValueError("GX 结构错误: 下跌特征序列 起点比终点低", level, start.dt, end.dt)

        elif (start.style == Shape.S) and (end.style == Shape.S):
            if start.l.low > end.r.high:
                raise ValueError("SS 结构错误: 上涨特征序列 起点比终点高", level, start.dt, end.dt)
        elif (start.style == Shape.X) and (end.style == Shape.X):
            if start.l.high < end.r.low:
                raise ValueError("XX 结构错误: 下跌特征序列 起点比终点低", level, start.dt, end.dt)

        elif (start.style == Shape.S) and (end.style == Shape.G):
            if start.l.low > end.high:
                raise ValueError("SG 结构错误: 上涨特征序列 起点比终点高", level, start.dt, end.dt)
        elif (start.style == Shape.X) and (end.style == Shape.D):
            if start.l.high < end.low:
                raise ValueError("XD 结构错误: 下跌特征序列 起点比终点低", level, start.dt, end.dt)

        else:
            raise TypeError(doubleRelation(start, end), start.style, end.style)

        self.__level = level # 0 为笔， 大于0，为段
        self.__start = start
        self.__end = end
        self.elements = elements
        self.index = index
        self.isFixed = isFixed

        self.isVisual = isVisual # 虚段标志

        if not self.elements:
            return

        if type(self.elements[0]) is ChanCandle:
            if len(self.elements) < 5 and isFixed == 1:
                print("ChanFeature 警告: 笔结构不完整。", len(self.elements), self.isFixed)

            if self.__start.m != self.elements[0]:raise ValueError(5)
            if self.__end.m != self.elements[-1]:
                print(self.__end.m, self.elements[-1])
                raise ValueError(6)
        else:
            if len(self.elements) < 3:
                print("ChanFeature 警告: 线段结构不完整。", len(self.elements), self.isFixed)

            if self.__start != self.elements[0].start:raise ValueError(5)
            if self.__end != self.elements[-1].end:raise ValueError(6)

    @property
    def LEVEL(self) -> int:
        return self.__level
    @property
    def POWER(self) -> float:
        return self.HIGH - self.LOW
    @property
    def macd(self)  -> float:
        return sum([abs(o.macd) for o in self.elements])

    @property
    def cklines(self) -> List[ChanCandle]:
        result = []
        if self.__level == 0:
            return self.elements
        else:
            for tz in self.elements:
                result.extend(tz.cklines)
        return result

    @property
    def rsq(self) -> float:
        """拟合优度 R SQuare

        :return:
        """
        y = [ck.end for ck in self.cklines]
        num = len(y)
        x = list(range(num))

        x_squred_sum = sum([x1 * x1 for x1 in x])
        xy_product_sum = sum([x[i] * y[i] for i in range(num)])

        x_sum = sum(x)
        y_sum = sum(y)
        delta = float(num * x_squred_sum - x_sum * x_sum)
        if delta == 0:
            return 0.0
        y_intercept = (1 / delta) * (x_squred_sum * y_sum - x_sum * xy_product_sum)
        slope = (1 / delta) * (num * xy_product_sum - x_sum * y_sum)

        y_mean = sum(y) / len(y)
        ss_tot = sum([(y1 - y_mean) * (y1 - y_mean) for y1 in y]) + 0.00001
        ss_err = sum([(y[i] - slope * x[i] - y_intercept) * (y[i] - slope * x[i] - y_intercept) for i in range(len(x))])
        rsq = 1 - ss_err / ss_tot

        #return rsq
        return round(rsq, 8)

    @property
    def start(self) -> ChanFenXing:
        return self.__start

    @property
    def high(self) -> ChanFenXing:
        start = self.__start
        end = self.__end
        if (start.style == Shape.D) and (end.style == Shape.G):
            if start.low > end.high:
                raise ValueError("DG 结构错误: 上涨特征序列 起点比终点高", start.dt, end.dt)
            #return self.start
            return self.end
        elif (start.style == Shape.G) and (end.style == Shape.D):
            if start.high < end.low:
                raise ValueError("GD 结构错误: 下跌特征序列 起点比终点低", start.dt, end.dt)
            return self.start
            #return self.end

        elif (start.style == Shape.D) and (end.style == Shape.S):
            if start.low > end.r.high:
                raise ValueError("DS 结构错误: 上涨特征序列 起点比终点高", start.dt, end.dt)
            #return self.start
            return self.end
        elif (start.style == Shape.G) and (end.style == Shape.X):
            if start.high < end.r.low:
                raise ValueError("GX 结构错误: 下跌特征序列 起点比终点低", start.dt, end.dt)
            return self.start
            #return self.end

        elif (start.style == Shape.S) and (end.style == Shape.S):
            if start.l.low > end.r.high:
                raise ValueError("SS 结构错误: 上涨特征序列 起点比终点高", start.dt, end.dt)
            #return self.start
            return self.end
        elif (start.style == Shape.X) and (end.style == Shape.X):
            if start.l.high < end.r.low:
                raise ValueError("XX 结构错误: 下跌特征序列 起点比终点低", start.dt, end.dt)
            return self.start
            #return self.end

        elif (start.style == Shape.S) and (end.style == Shape.G):
            if start.l.low > end.high:
                raise ValueError("SG 结构错误: 上涨特征序列 起点比终点高", start.dt, end.dt)
            #return self.start
            return self.end
        elif (start.style == Shape.X) and (end.style == Shape.D):
            if start.l.high < end.low:
                raise ValueError("XD 结构错误: 下跌特征序列 起点比终点低", start.dt, end.dt)
            return self.start
            #return self.end

        else:
            raise TypeError(doubleRelation(start, end), start.style, end.style)

    @property
    def HIGH(self) -> int:
        if self.high.style is Shape.S:
            return self.high.r.high
        return self.high.fx

    @property
    def low(self) -> ChanFenXing:
        start = self.__start
        end = self.__end
        if (start.style == Shape.D) and (end.style == Shape.G):
            if start.low > end.high:
                raise ValueError("DG 结构错误: 上涨特征序列 起点比终点高", start.dt, end.dt)
            return self.start
            #return self.end
        elif (start.style == Shape.G) and (end.style == Shape.D):
            if start.high < end.low:
                raise ValueError("GD 结构错误: 下跌特征序列 起点比终点低", start.dt, end.dt)
            #return self.start
            return self.end

        elif (start.style == Shape.D) and (end.style == Shape.S):
            if start.low > end.r.high:
                raise ValueError("DS 结构错误: 上涨特征序列 起点比终点高", start.dt, end.dt)
            return self.start
            #return self.end
        elif (start.style == Shape.G) and (end.style == Shape.X):
            if start.high < end.r.low:
                raise ValueError("GX 结构错误: 下跌特征序列 起点比终点低", start.dt, end.dt)
            #return self.start
            return self.end

        elif (start.style == Shape.S) and (end.style == Shape.S):
            if start.l.low > end.r.high:
                raise ValueError("SS 结构错误: 上涨特征序列 起点比终点高", start.dt, end.dt)
            return self.start
            #return self.end
        elif (start.style == Shape.X) and (end.style == Shape.X):
            if start.l.high < end.r.low:
                raise ValueError("XX 结构错误: 下跌特征序列 起点比终点低", start.dt, end.dt)
            #return self.start
            return self.end

        elif (start.style == Shape.S) and (end.style == Shape.G):
            if start.l.low > end.high:
                raise ValueError("SG 结构错误: 上涨特征序列 起点比终点高", start.dt, end.dt)
            return self.start
            #return self.end
        elif (start.style == Shape.X) and (end.style == Shape.D):
            if start.l.high < end.low:
                raise ValueError("XD 结构错误: 下跌特征序列 起点比终点低", start.dt, end.dt)
            #return self.start
            return self.end

        else:
            raise TypeError(doubleRelation(start, end), start.style, end.style)

    @property
    def LOW(self) -> int:
        if self.low.style is Shape.X:
            return self.low.r.low
        return self.low.fx

    @property
    def end(self) -> ChanFenXing:
        return self.__end

    @property
    def direction(self) -> Direction:
        start = self.__start
        end = self.__end
        if (start.style == Shape.D) and (end.style == Shape.G):
            if start.low > end.high:
                raise ValueError("DG 结构错误: 上涨特征序列 起点比终点高", start.dt, end.dt)
            return Direction.Up
        elif (start.style == Shape.G) and (end.style == Shape.D):
            if start.high < end.low:
                raise ValueError("GD 结构错误: 下跌特征序列 起点比终点低", start.dt, end.dt)
            return Direction.Down

        elif (start.style == Shape.D) and (end.style == Shape.S):
            if start.low > end.r.high:
                raise ValueError("DS 结构错误: 上涨特征序列 起点比终点高", start.dt, end.dt)
            return Direction.Up
        elif (start.style == Shape.G) and (end.style == Shape.X):
            if start.high < end.r.low:
                raise ValueError("GX 结构错误: 下跌特征序列 起点比终点低", start.dt, end.dt)
            return Direction.Down

        elif (start.style == Shape.S) and (end.style == Shape.S):
            if start.l.low > end.r.high:
                raise ValueError("SS 结构错误: 上涨特征序列 起点比终点高", start.dt, end.dt)
            return Direction.Up
        elif (start.style == Shape.X) and (end.style == Shape.X):
            if start.l.high < end.r.low:
                raise ValueError("XX 结构错误: 下跌特征序列 起点比终点低", start.dt, end.dt)
            return Direction.Down

        elif (start.style == Shape.S) and (end.style == Shape.G):
            if start.l.low > end.high:
                raise ValueError("SG 结构错误: 上涨特征序列 起点比终点高", start.dt, end.dt)
            return Direction.Up
        elif (start.style == Shape.X) and (end.style == Shape.D):
            if start.l.high < end.low:
                raise ValueError("XD 结构错误: 下跌特征序列 起点比终点低", start.dt, end.dt)
            return Direction.Down

        else:
            raise TypeError(doubleRelation(start, end), start.style, end.style)

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def mid(self) -> float:
        '''中间值'''
        return (self.HIGH + self.LOW) / 2

    def __getitem__(self, i):
        return self.elements[i]
    def __str__(self):
        if self.LEVEL == 0:
            return f"ChanBi('{self.symbol}', {self.LEVEL}, {self.high.high}, {self.low.low}, {self.direction}, {self.start.dt}, {self.end.dt}, {len(self.elements)}, {self.isFixed})"
        elif self.LEVEL == 1:
            return f"ChanDuan('{self.symbol}', {self.LEVEL}, {self.high.high}, {self.low.low}, {self.direction}, {self.start.dt}, {self.end.dt}, {len(self.elements)}, {self.isFixed})"
        return f"特征序列('{self.symbol}', {self.LEVEL}, {self.high.high}, {self.low.low}, {self.direction}, {self.start.dt}, {self.end.dt}, {len(self.elements)}, {self.isFixed})"

    def __repr__(self):
        if self.LEVEL == 0:
            return f"ChanBi('{self.symbol}', {self.LEVEL}, {self.high.high}, {self.low.low}, {self.direction}, {self.start.dt}, {self.end.dt}, {len(self.elements)}, {self.isFixed})"
        elif self.LEVEL == 1:
            return f"ChanDuan('{self.symbol}', {self.LEVEL}, {self.high.high}, {self.low.low}, {self.direction}, {self.start.dt}, {self.end.dt}, {len(self.elements)}, {self.isFixed})"
        return f"特征序列('{self.symbol}', {self.LEVEL}, {self.high.high}, {self.low.low}, {self.direction}, {self.start.dt}, {self.end.dt}, {len(self.elements)}, {self.isFixed})"

    def __eq__(self, other):
        return (isinstance(other, type(self))) and \
               ((self.start, self.end, self.elements) == (other.start, other.end, other.elements))

    def toPillar(self) -> ChanCandle:
        t = self.check()
        if t:
            raise ChanException(t, self)
        if self.direction is Direction.Up:
            pillar = ChanCandle("Pillar", self.start.low, self.end.high, [])
            assert not (pillar.direction != self.direction), ValueError("王德发")
            return pillar
        if self.direction is Direction.Down:
            pillar = ChanCandle("Pillar", self.start.high, self.end.low, [])
            assert not (pillar.direction != self.direction), ValueError("王德发")
            return pillar

    def isNext(self, feature: "ChanFeature") -> bool:
        if not (isinstance(feature, type(self))):raise ChanException
        if self.end is feature.start:
            return True
        return False

    def check(self) -> int:
        if not self.start:return 1
        if not self.end:return -1
        if not self.elements:return -2
        if not (self.start.style in (Shape.G, Shape.D, Shape.S, Shape.X)):return 2
        if not (self.end.style in (Shape.G, Shape.D, Shape.S, Shape.X)):return -2
        if self.start.style == self.end.style:return 3
        if self.start.dt > self.end.dt:return 4
        if type(self.elements[0]) is ChanCandle:
            if self.start.m != self.elements[0]:return 5
            if self.end.m != self.elements[-1]:return 6
            #if self.LEVEL != 0:return 7
            low = min(self.elements, key=lambda x:x.low).low
            high = max(self.elements, key=lambda x:x.high).high
            if self.direction is Direction.Up:
                if self.start.m.low != low:print(self.start.dt, "check 向上笔, 起点不是最低点")
                if self.end.m.high != high:print(self.start.dt, "check 向上笔, 终点不是最高点")
            if self.direction is Direction.Down:
                if self.start.m.high != high:print(self.start.dt, "check 向下笔, 起点不是最高点")
                if self.end.m.low != low:print(self.start.dt, "check 向下笔, 终点不是最低点")
        else:
            if self.start != self.elements[0].start:return 5
            if self.end != self.elements[-1].end:return 6
            #if self.LEVEL <= 0:return 7
        '''
        if self.LEVEL == 0:
            if not type(self.elements[0]) is ChanCandle:return 9
        else:
            #if not type(self.elements[0]) is Feature:return 10
            if self.elements[0].LEVEL > self.LEVEL:
                return 11
            if self.elements[0].LEVEL == self.LEVEL:
                return 12
        '''

        return 0

@enforce_types
def mergeFeature(l:ChanFeature, r:ChanFeature) -> ChanFeature:
        ''' 特征序列合并
        向下取高高， 反之取低低
        '''
        if l.direction != r.direction:
            raise ValueError("非同向无法合并", l.direction, r.direction)
        if l.start.style != r.start.style:
            raise ValueError("非相同分型无法合并, start", l.start.direction, r.start.direction)
        if l.end.style != r.end.style:
            raise ValueError("非相同分型无法合并, end", l.end.direction, r.end.direction)

        if l.symbol != r.symbol:
            raise ValueError("符号不同无法合并")
        if l.LEVEL != r.LEVEL:
            raise ValueError("等级不同无法合并")

        if l.direction is Direction.Up:
            # 向上笔为向下线段的特征序列， 取低低
            if l.LOW == r.LOW:
                print("警告: 向下线段 出现低点相同！！")
            high = min([l.high, r.high], key=lambda x:x.high)
            low = min([l.low, r.low], key=lambda x:x.low)
            start = low
            end = high

        elif l.direction is Direction.Down:
            if l.HIGH == r.HIGH:
                print("警告: 向上线段 出现高点相同！！")
            high = max([l.high, r.high], key=lambda x:x.high)
            low = max([l.low, r.low], key=lambda x:x.low)
            start = high
            end = low

        else:
            raise TypeError(l.direction, "!=", r.direction)

        if l.start == start:
            elements = [l.elements[0], r.elements[-1]]
        else:
            elements = [r.elements[0], l.elements[-1]]
        if l.LEVEL == 0:
            elements = [start.m, end.m]
        #print(elements)

        if start.style == end.style:
            raise ValueError("相同分型无法成笔")
        feature = ChanFeature(start, end, elements, level=l.LEVEL, isFixed = l.isFixed + r.isFixed)
        return feature

@enforce_types
class ChanZhongShu:
    __slots__ = ["symbol", "elements", "ok", "__level", "__l", "__m", "__r"]
    def __init__(self, symbol:str, left:[ChanCandle, ChanFeature], mid:[ChanCandle, ChanFeature], right:[ChanCandle, ChanFeature], level:int=0):
        '''三根k线重叠区间为最小中枢
        '''
        self.symbol = symbol
        self.__level = level
        if left.check():raise ChanException(left.check())
        if mid.check():raise ChanException(mid.check())
        if right.check():raise ChanException(right.check())
        if not left.isNext(mid):raise ChanException(f"等级{level}, 无法通过连续性检测", left.end, mid.start)
        if not mid.isNext(right):raise ChanException(f"等级{level}, 无法通过连续性检测", mid.end, right.start)

        self.__l, self.__m, self.__r = left, mid, right

        self.elements = []

        if doubleRelation(left.toPillar(), right.toPillar()) in (Direction.JumpUp, Direction.JumpDown):
            self.ok = False
        else:
            self.ok = True

    @property
    def l(self) -> ChanFeature:
        return self.__l

    @property
    def m(self) -> ChanFeature:
        return self.__m

    @property
    def r(self) -> ChanFeature:
        return self.__r

    @property
    def interval(self) -> ChanCandle:
        if self.m.direction is Direction.Up:
            return ChanCandle("Pillar", self.ZD, self.ZG, [])
        elif self.m.direction is Direction.Down:
            return ChanCandle("Pillar", self.ZG, self.ZD, [])
        else:
            raise ChanException("未知中枢区间")

    @property
    def LEVEL(self) -> int:
        return self.__level
    @property
    def status(self) -> bool:
        return self.DD < self.ZD < self.ZG < self.GG

    @property
    def direction(self) -> Direction:
        return self.m.direction

    @property
    def start(self) -> [ChanCandle, ChanFeature]:
        return self.l

    @property
    def end(self) -> [ChanCandle, ChanFeature]:
        if not self.elements:
            return self.r
        return self.elements[-1]

    @property
    def G(self) -> float:
        elements = [self.l, self.m, self.r]
        elements.extend(self.elements)
        return min(elements, key=lambda x:x.high.high).high.high

    @property
    def D(self) -> float:
        elements = [self.l, self.m, self.r]
        elements.extend(self.elements)
        return max(elements, key=lambda x:x.low.low).low.low

    @property
    def GG(self) -> float:
        elements = [self.l, self.m, self.r]
        elements.extend(self.elements)
        return max(elements, key=lambda x:x.high.high).high.high

    @property
    def DD(self) -> float:
        elements = [self.l, self.m, self.r]
        elements.extend(self.elements)
        return min(elements, key=lambda x:x.low.low).low.low

    @property
    def ZG(self) -> float:
        return min((self.l, self.m, self.r), key=lambda x:x.high.high).high.high

    @property
    def ZD(self) -> float:
        return max((self.l, self.m, self.r), key=lambda x:x.low.low).low.low

    def __str__(self):
        return f"ChanZhongShu('{self.symbol}', {self.ZG}, {self.ZD}, {self.direction}, {len(self.elements)})"

    def __repr__(self):
        return f"ChanZhongShu('{self.symbol}', {self.ZG}, {self.ZD}, {self.direction}, {len(self.elements)})"

    @enforce_types
    def isNext(self, feature: [ChanCandle, ChanFeature]) -> bool:
        if not self.elements:
            return self.r.isNext(feature)
        return self.elements[-1].isNext(feature)

    @enforce_types
    def inInterval(self, feature: [ChanCandle, ChanFeature]) -> bool:
        relation = doubleRelation(self.interval, feature.toPillar())
        if relation in (Direction.JumpUp, Direction.JumpDown):
            return False
        return True

    def toPillar(self):
        if self.m.direction is Direction.Up:
            return ChanCandle("Pillar", self.DD, self.GG, [])
        elif self.m.direction is Direction.Down:
            return ChanCandle("Pillar", self.GG, self.DD, [])
        else:
            raise ChanException("未知中枢区间")

    def isOutspread(self, zs:"ChanZhongShu") -> bool:
        # 是否扩张
        relation = doubleRelation(self.interval, zs.interval)
        if relation == Direction.JumpUp:
            if self.GG >= zs.DD:
                return True
        elif relation == Direction.JumpDown:
            if self.DD >= zs.GG:
                return True
        else:
            print("中枢是否扩展关系", relation)

    def isExtension(self, zs:"ChanZhongShu") -> bool:
        # 是否扩展
        ...

    @enforce_types
    def pop(self, feature: [ChanCandle, ChanFeature]) -> [ChanCandle, ChanFeature]:
        if not self.elements:
            self.ok = False
            tz = self.__r
            self.__r = None
            return tz
        else:
            return self.elements.pop()

    @enforce_types
    def add(self, feature: [ChanCandle, ChanFeature]) -> int:
        if not feature:
            raise ValueError("无效特征序列")
        if not self.isNext(feature):
            print("ChanZhongShu 首尾 不呼应")
            return 3

        if not self.ok:
            print("... 不是中枢 ...")
            return 1

        if not self.inInterval(feature):
            return 4
        self.elements.append(feature)
        return 0


@enforce_types
def hasInclude(left:ChanCandle, right:[RawCandle, ChanCandle]) -> Direction:
    if (left.low >= right.low) and (left.high <= right.high):
        return Direction.Right # "右" # 右包左, 逆序
    if (left.low <= right.low) and (left.high >= right.high):
        return Direction.Left # "左" # 左包右, 顺序

@enforce_types
def doubleRelation(left:ChanCandle, right:ChanCandle) -> Direction:
    '''
        两棵k线的所有关系
    '''
    if left == right:
        raise ChanException("相同对象无法比较")
    if 1:
        if (left.low <= right.low) and (left.high >= right.high):
            return Direction.Left # "左包右" # 顺序

        if (left.low >= right.low) and (left.high <= right.high):
            return Direction.Right # "右包左" # 逆序

        if (left.low < right.low) and (left.high < right.high):
            if left.high < right.low:
                return Direction.JumpUp # "跳涨"
            return Direction.Up # "上涨"

        if (left.low >= right.low) and (left.high >= right.high):
            if left.low > right.high:
                return Direction.JumpDown # "跳跌"
            return Direction.Down # "下跌"

@enforce_types
def tripleRelation(l:ChanCandle, m:ChanCandle, r:ChanCandle, isRight=False) -> Shape:
    '''
        三棵缠论k线的所有关系#, 允许逆序包含存在。
        顶分型: 中间高点为三棵最高点。
        底分型: 中间低点为三棵最低点。
        上升分型: 高点从左至右依次升高
        下降分型: 低点从左至右依次降低
        # 喇叭口型: 高低点从左至右依次更高更低

        优先识别顶、底分型。
    '''
    if any((l == m, m == r, l == r)):
        raise ChanException("相同对象无法比较")

    if 1:
        lm = doubleRelation(l, m)
        mr = doubleRelation(m, r)

        if lm in (Direction.Up, Direction.JumpUp):
            # 涨
            if mr in (Direction.Up, Direction.JumpUp):
                # 涨
                return Shape.S
            if mr in (Direction.Down, Direction.JumpDown):
                # 跌
                return Shape.G
            if mr is Direction.Left:
                # 顺序包含
                raise ValueError("顺序包含 mr")
            if mr is Direction.Right and isRight:
                # 逆序包含
                return Shape.S

        if lm in (Direction.Down, Direction.JumpDown):
            # 跌
            if mr in (Direction.Up, Direction.JumpUp):
                # 涨
                return Shape.D
            if mr in (Direction.Down, Direction.JumpDown):
                # 跌
                return Shape.X
            if mr is Direction.Left:
                # 顺序包含
                raise ValueError("顺序包含 mr")
            if mr is Direction.Right and isRight:
                # 逆序包含
                return Shape.X

        if lm is Direction.Left:
            # 顺序包含
            raise ValueError("顺序包含 lm")
        if lm is Direction.Right and isRight:
            # 逆序包含
            if mr in (Direction.Up, Direction.JumpUp):
                # 涨
                return Shape.D
            if mr in (Direction.Down, Direction.JumpDown):
                # 跌
                return Shape.G
            if mr is Direction.Left:
                # 顺序包含
                raise ValueError("顺序包含 mr")
            if mr is Direction.Right and isRight:
                # 逆序包含
                return Shape.T # 喇叭口型


@enforce_types
def removeInclude(ck:ChanCandle, raw:RawCandle, direction:Direction) -> ChanCandle:
    elements = ck.elements
    elements.append(raw)
    if direction in (Direction.Down, Direction.JumpDown):
        high = min(ck.high, raw.high)
        low = min(ck.low, raw.low)
    elif direction in (Direction.Up, Direction.JumpUp):
        low = max(ck.low, raw.low)
        high = max(ck.high, raw.high)
    else:
        raise ChanException

    if ck.start > ck.end:
        start = high
        end = low
    else:
        start = low
        end = high
    return ChanCandle(raw.symbol, start, end, elements)

@enforce_types
def handleInclude(l:ChanCandle, r:ChanCandle, raw:RawCandle):
    if hasInclude(l,r):raise ChanException("输入数据存在包含关系")
    if not hasInclude(r, raw):
        return False, raw.toChanCandle()
    return True, removeInclude(r, raw, doubleRelation(l, r))

@enforce_types
class KlineHandler:
    ''' K线 去除包含处理 '''
    def __init__(self, klines:List[RawCandle]=[], cklines:List[ChanCandle]=[]):
        self.klines:List[RawCandle] = klines
        self.cklines:List[ChanCandle] = cklines

    @property
    def step(self) -> datetime:
        '''周期'''
        if len(self.klines) >= 2:
            return self.klines[1].dt - self.klines[0].dt
        else:
            ...#self.__times.append(k.dt)

    def calcBOLL(self, timeperiod=20, std=2):
        # https://blog.csdn.net/qq_41437512/article/details/105473845
        # https://wiki.mbalib.com/wiki/%E5%B8%83%E6%9E%97%E7%BA%BF%E6%8C%87%E6%A0%87

        self.calcMA(timeperiod)
        size = len(self.klines)
        if size < timeperiod:return
        last = self.klines[-1]
        second = self.klines[-2]

        md = math.sqrt( ((last.close - getattr(last, f"ma{timeperiod}")) ** 2) / timeperiod)
        mb = getattr(second, f"ma{timeperiod}")
        up = mb + std*md
        dn = mb - std*md
        setattr(last, f"boll{timeperiod}", {"up":up, "mid": mb, "dn":dn})

    def calcKDJ(self, timeperiod=9, a=2/3, b=1/3):
        # https://wiki.mbalib.com/wiki/%E9%9A%8F%E6%9C%BA%E6%8C%87%E6%A0%87
        size = len(self.klines)
        if size < timeperiod:return
        last = self.klines[-1]
        second = self.klines[-2]

        l = min(self.klines[-timeperiod:], key=lambda x:x.low).low
        h = max(self.klines[-timeperiod:], key=lambda x:x.high).high
        n = h-l
        if n == 0:
            #print(self.klines[-timeperiod:])
            n = 1
            print(colored("float division by zero", "red"))
            config.logger.warning("float division by zero")
        rsv = ((last.close-l) / n) * 100
        last.K = a*second.K + b*rsv
        last.D = a*second.D + b*last.K
        last.J = 3*last.D - 2*last.K

    def calcEMA(self, timeperiod=5):
        if len(self.klines) == 1:
            ema = self.klines[-1].close
        else:
            ema = (2 * self.klines[-1].close + getattr(self.klines[-2], f"ema{timeperiod}") * (timeperiod-1)) / (timeperiod+1)
        setattr(self.klines[-1], f"ema{timeperiod}", ema)

    def calcMA(self, timeperiod=5):
        if len(self.klines) < timeperiod:
            ma = self.klines[-1].close
        else:
            ma = sum([k.close for k in self.klines[-timeperiod:]]) / timeperiod
        setattr(self.klines[-1], f"ma{timeperiod}", ma)

    def calcMACD(self, fastperiod=12, slowperiod=26, signalperiod=9):
        self.calcEMA(fastperiod)
        self.calcEMA(slowperiod)
        DIF = getattr(self.klines[-1], f"ema{fastperiod}") - getattr(self.klines[-1], f"ema{slowperiod}")
        self.klines[-1].DIF = DIF

        if len(self.klines) == 1:
            dea = self.klines[-1].DIF
        else:
            dea = (2 * self.klines[-1].DIF + self.klines[-2].DEA * (signalperiod-1)) / (signalperiod+1)

        self.klines[-1].DEA = dea
        self.klines[-1].macd = (DIF - dea) * 2

    def calc1355(self):
        self.calcEMA(13)
        self.calcEMA(55)
        self.calcEMA(220)
        self.calcEMA(576)
        self.calcEMA(676)

    @enforce_types
    def append(self, k:RawCandle):
        if k in self.klines[-5:]:
            print(self.__class__.__name__, colored("重复k线", "red"))
            return self

        if len(self.klines) > 1:
            if self.klines[-1].dt > k.dt:
                raise ValueError("时序逆序", self.klines[-1].dt, k.dt)
            step = k.dt - self.klines[-1].dt
            if step > self.step:
                print("### 数据跳跃", step)

        self.klines.append(k)
        self.klines[-1].index = len(self.klines)+1
        self.calcMACD()
        self.calcBOLL()
        self.calcKDJ()
        hasReplaced = False
        if not self.cklines:
            ck = k.toChanCandle()
        elif len(self.cklines) == 1:
            if hasInclude(self.cklines[-1], k):
                ck = removeInclude(self.cklines[-1], k, Direction.Up)
                hasReplaced = True
            else:
                ck = k.toChanCandle()
        else:
            hasReplaced, ck = handleInclude(self.cklines[-2], self.cklines[-1], k)

        if hasReplaced:
            self.cklines[-1] = ck
        else:
            self.cklines.append(ck)
        self.cklines[-1].index = len(self.cklines) - 1

        return hasReplaced, ck

    def pop(self):
        k = self.klines.pop()
        elements = self.cklines[-1].elements
        size = len(elements)
        hasReplaced = False
        if size == 1:
            ck = self.cklines.pop()
        elif size == 2:
            ck = elements[0].toChanCandle()
            self.cklines[-1] = ck
        else:
            ck = elements[0].toChanCandle()
            i = 1
            size -= 1
            if len(self.cklines) >= 2:
                while size:
                    k = elements[i]
                    hasReplaced, ck = handleInclude(self.cklines[-2], ck, k)
                    if not hasReplaced:
                        raise ChanException
                    size -= 1
                    i += 1
            else:
                while size:
                    k = elements[i]
                    ck = removeInclude(ck, k, Direction.Up)
                    size -= 1
                    i += 1
            hasReplaced = True
            self.cklines[-1] = ck
        return hasReplaced, ck

class FenXingHandler(FeatureMachine):
    ''' 分型处理 '''
    def __init__(self):
        super(FenXingHandler, self).__init__()
        self.shapeMachine = Machine([0,-1,1]) # -1, 阴, 0, 混沌, 1, 阳
        self.stack = Stack()
        deal = self._deal_left, self._deal_mid, self._deal_right
        state = self.shapeMachine.state
        candles = self._left, self._mid, self._right
        self.stack.push((deal, state, candles, None))

    def pop(self, force=0):
        # 回退
        if force:
            return True
        #print("堆栈大小", len(self.stack.stack))
        deal, state, candles, style = self.stack.pop()
        #print("# do backed start", deal, state, candles)

        deal, state, candles, style = self.stack.gettop()
        self._deal_left, self._deal_mid, self._deal_right = deal
        self.shapeMachine.state = state
        self._left, self._mid, self._right = candles
        #print("= do backed end", deal, state, candles)
        if style:
            return True
        return False

    def append(self, ck:ChanCandle, SX=1, right=0):
        # 前进
        fx = None
        deal = self._deal_left, self._deal_mid, self._deal_right
        state = self.shapeMachine.state
        #print(deal, state)
        #SX = 1 # 是否需要返回 上升分型下降分型
        #right = 0 # 是否允许逆序包含
        if self.shapeMachine.state == 0:
            if self._deal_left:
                deal = (0,1,0)
                self.setLeft(ck, sys._getframe().f_lineno) # self._left = ck

            elif self._deal_mid:
                # 第二特征序列
                relation = doubleRelation(self._left, ck)
                if relation in (Direction.Up, Direction.JumpUp):
                    # 涨
                    # self.log(sys._getframe().f_lineno, "第二特征序列", relation)
                    deal = (0,0,1)
                    self.setMid(ck, sys._getframe().f_lineno)
                    state = 1

                elif relation in (Direction.Down, Direction.JumpDown):
                    # 跌
                    # self.log(sys._getframe().f_lineno, "第二特征序列", relation)
                    deal = (0,0,1)
                    self.setMid(ck, sys._getframe().f_lineno)
                    state = -1
                else:
                    raise ChanException("存在包含关系", relation)

        elif self.shapeMachine.state == 1:#self.machine.is_阳():
            if self._deal_left:
                deal = (0,1,0)
                self.setLeft(ck, sys._getframe().f_lineno) # self._left = ck

            elif self._deal_mid:
                # 第二特征序列
                relation = doubleRelation(self._left, ck)
                if relation in (Direction.Up, Direction.JumpUp):
                    # 涨
                    # self.log(sys._getframe().f_lineno, "第二特征序列", relation)
                    deal = (0,0,1)
                    self.setMid(ck, sys._getframe().f_lineno)

                elif relation in (Direction.Down, Direction.JumpDown):
                    # 跌
                    # self.log(sys._getframe().f_lineno, "第二特征序列", relation)
                    deal = (0,0,1)
                    self.setMid(ck, sys._getframe().f_lineno)
                    state = -1
                elif relation in (Direction.Left, ):
                    # 顺序包含
                    # self.log(sys._getframe().f_lineno, "第二特征序列", relation)
                    raise ValueError("顺序包含", relation, self._left.index, ck.index)
                elif relation in (Direction.Right, ):
                    # 逆序包含
                    # self.log(sys._getframe().f_lineno, "第二特征序列", relation)
                    if not right:raise ValueError("逆序包含", relation, self._left.index, ck.index)
                    deal = (0,0,1)
                    self.setMid(ck, sys._getframe().f_lineno)
                else:
                    raise TypeError(relation)

            elif self._deal_right:
                # 第三特征序列
                relation = doubleRelation(self._mid, ck)
                if relation in (Direction.Up, Direction.JumpUp):
                    # 涨
                    # self.log(sys._getframe().f_lineno, "第三特征序列", relation)
                    if SX:fx = ChanFenXing(self._left, self._mid, ck)
                    deal = (0,0,1)
                    self.setLeft(self._mid, sys._getframe().f_lineno)
                    self.setMid(ck, sys._getframe().f_lineno)

                elif relation in (Direction.Down, Direction.JumpDown):
                    # 下跌， 顶分型成立
                    # self.log(sys._getframe().f_lineno, "第三特征序列", relation)
                    fx = ChanFenXing(self._left, self._mid, ck)
                    deal = (0,0,1)
                    self.setLeft(self._mid, sys._getframe().f_lineno)
                    self.setMid(ck, sys._getframe().f_lineno)
                    state = -1

                elif relation in (Direction.Left, ):
                    # 顺序包含
                    # self.log(sys._getframe().f_lineno, "第三特征序列", relation)
                    raise ValueError("顺序包含", relation, self._mid.index, ck.index)
                elif relation in (Direction.Right, ):
                    # 逆序包含
                    # self.log(sys._getframe().f_lineno, "第三特征序列", relation)
                    if not right:raise ValueError("逆序包含", relation, self._left.index, ck.index)
                    if SX:fx = ChanFenXing(self._left, self._mid, ck)
                    deal = (0,0,1)
                    self.setLeft(self._mid, sys._getframe().f_lineno)
                    self.setMid(ck, sys._getframe().f_lineno)

                else:
                    raise TypeError(relation)

        elif self.shapeMachine.state == -1:
            if self._deal_left:
                deal = (0,1,0)
                self.setLeft(ck, sys._getframe().f_lineno)

            elif self._deal_mid:
                # 第二特征序列
                relation = doubleRelation(self._left, ck)
                if relation in (Direction.Up, Direction.JumpUp):
                    # 涨
                    # self.log(sys._getframe().f_lineno, "第二特征序列", relation)
                    deal = (0,0,1)
                    self.setMid(ck, sys._getframe().f_lineno)
                    state = 1

                elif relation in (Direction.Down, Direction.JumpDown):
                    # 跌
                    # self.log(sys._getframe().f_lineno, "第二特征序列", relation)
                    deal = (0,0,1)
                    self.setMid(ck, sys._getframe().f_lineno)

                elif relation in (Direction.Left, ):
                    # 顺序包含
                    # self.log(sys._getframe().f_lineno, "第二特征序列", relation)
                    raise ValueError("顺序包含", relation, self._left, ck)
                elif relation in (Direction.Right, ):
                    # 逆序包含
                    # self.log(sys._getframe().f_lineno, "第二特征序列", relation)
                    if not right:raise ValueError("逆序包含", relation, self._left.index, ck.index)
                    deal = (0,0,1)
                    self.setMid(ck, sys._getframe().f_lineno)
                else:
                    raise TypeError(relation)
                state

            elif self._deal_right:
                # 第三特征序列
                relation = doubleRelation(self._mid, ck)
                if relation in (Direction.Up, Direction.JumpUp):
                    # 涨, 底分型成立
                    # self.log(sys._getframe().f_lineno, "第三特征序列", relation)
                    fx = ChanFenXing(self._left, self._mid, ck)
                    deal = (0,0,1)
                    self.setLeft(self._mid, sys._getframe().f_lineno)
                    self.setMid(ck, sys._getframe().f_lineno)
                    state = 1

                elif relation in (Direction.Down, Direction.JumpDown):
                    # 跌
                    # self.log(sys._getframe().f_lineno, "第三特征序列", relation)
                    if SX:fx = ChanFenXing(self._left, self._mid, ck)
                    deal = (0,0,1)
                    self.setLeft(self._mid, sys._getframe().f_lineno)
                    self.setMid(ck, sys._getframe().f_lineno)

                elif relation in (Direction.Left, ):
                    # 顺序包含
                    # self.log(sys._getframe().f_lineno, "第三特征序列", relation)
                    raise ValueError("顺序包含", relation, self._mid, ck)

                elif relation in (Direction.Right, ):
                    # 逆序包含
                    # self.log(sys._getframe().f_lineno, "第三特征序列", relation)
                    if not right:raise ValueError("逆序包含", relation, self._left.index, ck.index)
                    if SX:fx = ChanFenXing(self._left, self._mid, ck)
                    deal = (0,0,1)
                    self.setLeft(self._mid, sys._getframe().f_lineno)
                    self.setMid(ck, sys._getframe().f_lineno)
                else:
                    raise TypeError(relation)

        self._deal_left, self._deal_mid, self._deal_right = deal
        self.shapeMachine.state = state
        candles = self._left, self._mid, self._right
        self.stack.push((deal, state, candles, fx.style if fx else None))
        return fx

class BiHandler:
    ''' 笔 处理 '''

    @classmethod
    def generator(cls, points: list[int]):
        result = []
        size = len(points)
        i = 0
        pair = "generator"
        dt = datetime(2021, 9, 3, 19, 50, 40, 916152)
        while i+1 < size:
            s = points[i]
            e = points[i+1]
            m = abs(s-e) * 0.2

            h = s
            n = 0
            while n < 5:
                if s > e:
                    # down
                    open = h
                    high = h
                    h = round(h-m,4)
                    low = h
                    close = h
                    result.append(RawCandle(pair, dt, float(open), float(high), float(low), float(close), 1.0))
                    dt = datetime.fromtimestamp(dt.timestamp()+60*60)
                else:
                    # up
                    low = h
                    close = h
                    h = round(h+m,4)
                    open = h
                    high = h
                    result.append(RawCandle(pair, dt, float(open), float(high), float(low), float(close), 1.0))
                    dt = datetime.fromtimestamp(dt.timestamp()+60*60)
                n += 1
            i += 1
        return result

    def __init__(self, cklines:List[ChanCandle], zsh=None):
        self.cklines:List[ChanCandle] = cklines
        self.points:List[ChanFenXing] = []
        self.features:List[ChanFeature] = []
        self.biMachine = Machine([0,-1,1,-2,2]) # -2, 太阴, -1, 少阴, 0, 混沌, 1, 少阳, 2, 太阳

        self.hasUpdated = False
        self.hasBacked = False
        self.hasReplaced = False
        self.poped = 0
        self.appended = 0

        self.B = None
        self.T = None

        self.stack = Stack()

        self.stateStack = Stack()
        self.stateStack.push(self.biMachine.state)

        self.zsh = zsh
        self.lastFx = None

    def log(self, *args, **kwords):
        if config.debug:
            print("+", self.LEVEL, self.__class__.__name__, *args, **kwords)
            return
        if config.debug:
            config.logger.debug(self.__class__.__name__, *args, **kwords)

    @property
    def LEVEL(self) -> int:
        return 0
    @property
    def state(self):
        return {0:"混沌", 1:"少阳", 2:"老阳", -1:"少阴", -2:"老阴"} [self.biMachine.state]

    def checkSegments(self):
        """ 检测每笔 是否有BUG """
        count = 0
        for feature in self.features:
            if feature.direction is Direction.Up:
                if not (feature.start.m.low == min(feature.elements, key=lambda x:x.low).low):
                    print( 1, count)
                if not (feature.end.m.high == max(feature.elements, key=lambda x:x.high).high):
                    print( 2, count)
            else:
                if not (feature.start.m.high == max(feature.elements, key=lambda x:x.high).high):
                    print( 3, count)
                if not (feature.end.m.low == min(feature.elements, key=lambda x:x.low).low):
                    print( 4, count)
            count += 1

    @enforce_types
    def __replace_bi(self, fx:ChanFenXing, n:int):
        ck, old = self.__pop_bi(n)
        if old and old.end.style != fx.style:raise ChanException("分型不同无法替换")
        new = self.__append_bi(fx, n)
        if self.hasBacked and self.hasUpdated:
            self.log("line:", n, colored("__replaced_bi is called", "blue"), old == new)
            self.hasReplaced = True
            self.hasBacked  = False
            self.hasUpdated = False
        return old,new

    @enforce_types
    def __append_bi(self, fx: ChanFenXing, n:int):
        if fx.style in (Shape.G, Shape.D):
            self.log("line:", n, colored("__append_bi is called", "red"), fx.m)
            if self.points:
                if self.points[-1].style == fx.style:
                    raise TypeError("无法衔接")
            self.points.append(fx)

            if len(self.points) >= 2:
                start = self.points[-2]
                end = self.points[-1]
                bi = ChanFeature(start, end, self.cklines[start.m.index:end.m.index+1], level=0)

                if self.features:
                    if not self.features[-1].isNext(bi):
                        raise ValueError("无法连接", self.features[-1].end, bi.start, len(self.features))
                if len(self.features) >= 2:
                    if not self.features[-2].isNext(self.features[-1]):
                        raise ValueError("append 无法衔接")

                assert not bi.check()
                self.features.append(bi)
                self.features[-1].index = len(self.features)
                self.hasUpdated = True
                self.appended += 1
                self.log("line:", n, colored("__append_bi is called", "red"), bi)

                if config.debug:
                    low = min(bi.elements, key=lambda x:x.low)
                    high = max(bi.elements, key=lambda x:x.high)
                    HIGH = high.high
                    LOW = low.low
                    if bi.direction is Direction.Up:
                        if start.m.low != LOW:
                            print(bi.start.dt, "向上笔, 起点不是最低点", "\nlow:", low, "\nstart:", start.m)
                            #raise Exception
                        if end.m.high != HIGH:
                            print(bi.start.dt, "向上笔, 终点不是最高点", "\nhigh:", high, "\nend:", end.m)
                            #raise Exception
                    if bi.direction is Direction.Down:
                        if start.m.high != HIGH:
                            print(bi.start.dt, "向下笔, 起点不是最高点", "\nhigh:", high, "\nstart:", start.m)
                            #raise Exception
                        if end.m.low != LOW:
                            print(bi.start.dt, "向下笔, 终点不是最低点", "\nlow:", low, "\nend:", end.m)
                            #raise Exception

                # 处理中枢
                if self.zsh:
                    self.zsh.append(bi)
                return bi

        else:
            raise TypeError(ck.style, "分型错误")

    def __pop_bi(self, n:int):
        if self.points:
            fx = self.points.pop()
            tz = None
            if self.features:
                tz = self.features.pop()
                self.poped += 1
                self.hasBacked = True
                if tz.end != fx:
                    raise ChanException("王德发")

                # 开始 处理中枢
                if self.zsh:
                    t = self.zsh.pop(tz, sys._getframe().f_lineno)
                # 结束 中枢处理

            if len(self.features) >= 2:
                if self.features[-2].end != self.features[-1].start:
                    raise ValueError("pop 无法衔接")
            self.log("line:", n, colored("__pop_bi is called, fx =", "green"), fx)
            self.log("line:", n, colored("__pop_bi is called, tz =", "green"), tz)
            return fx, tz

    @enforce_types
    def getLow(self, l: ChanCandle, r: ChanCandle):
        return min(self.cklines[l.index:r.index], key=lambda x:x.low)

    @enforce_types
    def getHigh(self, l: ChanCandle, r: ChanCandle):
        return max(self.cklines[l.index:r.index], key=lambda x:x.high)

    def checkPoints(self, fx):
        states = {-1: Shape.G, -2: Shape.D, 1:Shape.D, 2:Shape.G}
        if not (self.points[-1].style is states[self.biMachine.state]):
            raise ChanException(self.state, "状态不符", self.points[-1].style, "last style:", fx.style, "应为:", states[self.biMachine.state])

    def pop(self):
        if self.points:
            ck = self.cklines[-1]
            fx = self.points[-1]
            elements = fx.m.elements
            if set(elements) & set(ck.elements):
                self.__pop_bi(sys._getframe().f_lineno)
                self.stateStack.pop()
                self.biMachine.state = self.stateStack.gettop()

    @enforce_types
    def append(self, fx: ChanFenXing):
        self.hasUpdated = False
        self.hasBacked = False
        self.hasReplaced = False
        self.poped = 0
        self.appended = 0

        self.log("line:", sys._getframe().f_lineno, "APPEND", self.points[-1].style if self.points else "没有数据", fx.style, self.state, fx.m)

        state = self.biMachine.state
        ps = len(self.points)

        force = 0
        isNew = 1
        #pdb.set_trace()
        info = ""
        while 1:
            if not self.points:
                self.biMachine.state = 0
            elif self.points[-1].m is fx.m:
                if self.points[-1].style == fx.style:
                    self.__replace_bi(fx, sys._getframe().f_lineno)

            if self.points:
                self.checkPoints(fx)

            match (self.biMachine.state, fx.style):
                case 0, Shape.X: # 混沌
                    break

                case 0, Shape.D: # 混沌
                    self.__append_bi(fx, sys._getframe().f_lineno)
                    state = -2
                    break

                case 0, Shape.S: # 混沌
                    break

                case 0, Shape.G: # 混沌
                    self.__append_bi(fx, sys._getframe().f_lineno)
                    state = 2
                    break

                case 1, Shape.X: # 少阳
                    if fx.r.low < self.points[-1].low:
                        self.__pop_bi(sys._getframe().f_lineno)
                        state = -1
                        # 人工修正
                        tmp = self.T
                        if tmp and self.points[-1].m.dt < tmp.m.dt and tmp.m.high > self.points[-1].m.high:
                            self.__replace_bi(tmp, sys._getframe().f_lineno)
                            bi = self.cklines[self.points[-1].m.index:fx.m.index+1]
                            if len(bi) >= 5:
                                state = -1
                            else:
                                state = 2
                            info = "修正1X"
                    break

                case 1, Shape.D: # 少阳
                    if fx.m.low < self.points[-1].m.low:
                        self.__replace_bi(fx, sys._getframe().f_lineno)
                        state = -2
                    break

                case 1, Shape.S: # 少阳
                    break

                case 1, Shape.G: # 少阳
                    if ps >= 2:
                        if fx.m is self.getHigh(self.points[-1].m, fx.r):
                            self.T = fx
                            info ="1G"

                    relation = doubleRelation(self.points[-1].m, fx.m)
                    if relation in (Direction.Up, Direction.JumpUp):
                        # 是否满足下一状态的条件 1 → 2 → -1 → -2 → 1
                        bi = self.cklines[self.points[-1].m.index:fx.m.index+1]
                        if bi[0] != self.points[-1].m or bi[-1] != fx.m:raise Exception
                        if len(bi) < 5:
                            #print("1G 不满足笔添加")
                            break
                        if force:
                            self.__append_bi(fx, sys._getframe().f_lineno)
                            state = 2
                            break
                        high = max(bi, key=lambda x:x.high)
                        low = min(bi, key=lambda x:x.low)
                        if (low,high) == (self.points[-1].m, fx.m):
                            self.__append_bi(fx, sys._getframe().f_lineno)
                            state = 2
                    else:
                        if fx.r.low < self.points[-1].m.low:
                            self.__pop_bi(sys._getframe().f_lineno)
                            state = -1
                    break

                case 2, Shape.X: # 太阳
                    size = (fx.r.index - self.points[-1].m.index)
                    if size >= 4:
                        # 是否满足下一状态的条件 1 → 2 → -1 → -2 → 1
                        state = -1
                    break

                case 2, Shape.D: # 太阳
                    if ps >= 2:
                        if fx.m is self.getLow(self.points[-1].m, fx.r):
                            self.B = fx
                            info = "2D"

                    size = (fx.r.index - self.points[-1].m.index)
                    relation = doubleRelation(self.points[-1].m, fx.m)
                    if size >= 4:
                        # 是否满足下一状态的条件 1 → 2 → -1 → -2 → 1
                        if fx.m.index - self.points[-1].m.index >= 4 and relation in (Direction.Down, Direction.JumpDown):
                            # 是否满足成笔的条件
                            bi = self.cklines[self.points[-1].m.index:fx.m.index+1]
                            if bi[0] != self.points[-1].m or bi[-1] != fx.m:raise Exception
                            if len(bi) < 5:raise Exception
                            if force:
                                self.__append_bi(fx, sys._getframe().f_lineno)
                                state = -2
                                break
                            high = max(bi, key=lambda x:x.high)
                            low = min(bi, key=lambda x:x.low)
                            if (high,low) == (self.points[-1].m, fx.m):
                                self.__append_bi(fx, sys._getframe().f_lineno)
                                state = -2
                            else:
                                state = -1
                        else:
                            state = -1
                    else:
                        if fx.r.high > self.points[-1].m.high:
                            self.__pop_bi(sys._getframe().f_lineno)
                            if self.points and fx.m is self.getLow(self.points[-1].m, fx.r):
                                self.__replace_bi(fx, sys._getframe().f_lineno)
                                state = -2
                            else:
                                state = 1
                    break

                case 2, Shape.S: # 太阳
                    size = (fx.r.index - self.points[-1].m.index)
                    if size >= 4:
                        # 是否满足下一状态的条件 1 → 2 → -1 → -2 → 1
                        if fx.r.high > self.points[-1].high:
                            self.__pop_bi(sys._getframe().f_lineno)
                            state = 1
                        else:
                            state = -1
                    else:
                        if fx.r.high > self.points[-1].high:
                            self.__pop_bi(sys._getframe().f_lineno)
                            state = 1
                    break

                case 2, Shape.G: # 太阳
                    size = (fx.r.index - self.points[-1].m.index)
                    if size >= 4:
                        # 是否满足下一状态的条件 1 → 2 → -1 → -2 → 1
                        if fx.m.high >= self.points[-1].m.high:
                            self.__replace_bi(fx, sys._getframe().f_lineno)
                            state = 2
                        else:
                            state = -1
                    else:
                        if fx.m.high >= self.points[-1].m.high:
                            self.__replace_bi(fx, sys._getframe().f_lineno)
                            state = 2
                    break

                case -1, Shape.X: # 少阴
                    break

                case -1, Shape.D: # 少阴
                    if ps >= 2:
                        if fx.m is self.getLow(self.points[-1].m, fx.r):
                            self.B = fx
                            info = "-1D"

                    relation = doubleRelation(self.points[-1].m, fx.m)
                    if relation in (Direction.Down, Direction.JumpDown):
                        # 是否满足下一状态的条件 1 → 2 → -1 → -2 → 1
                        bi = self.cklines[self.points[-1].m.index:fx.m.index+1]
                        if bi[0] != self.points[-1].m or bi[-1] != fx.m:raise Exception
                        if len(bi) < 5:
                            #print("-1D 不满足笔添加")
                            break
                        if force:
                            self.__append_bi(fx, sys._getframe().f_lineno)
                            state = -2
                            break
                        high = max(bi, key=lambda x:x.high)
                        low = min(bi, key=lambda x:x.low)
                        if (high,low) == (self.points[-1].m, fx.m):
                            self.__append_bi(fx, sys._getframe().f_lineno)
                            state = -2
                    else:
                        if fx.r.high > self.points[-1].m.high:
                            self.__pop_bi(sys._getframe().f_lineno)
                            state = 1
                    break

                case -1, Shape.S: # 少阴
                    if fx.r.high > self.points[-1].high:
                        self.__pop_bi(sys._getframe().f_lineno)
                        state = 1
                        # 人工修正
                        tmp = self.B
                        if tmp and self.points[-1].m.dt < tmp.m.dt and tmp.m.low < self.points[-1].m.low:
                            self.__replace_bi(tmp, sys._getframe().f_lineno)
                            bi = self.cklines[self.points[-1].m.index:fx.m.index+1]
                            if len(bi) >= 5:
                                state = 1
                            else:
                                state = -2
                            info = "修正-1S"
                    break

                case -1, Shape.G: # 少阴
                    if fx.m.high > self.points[-1].m.high:
                        self.__replace_bi(fx, sys._getframe().f_lineno)
                        state = 2
                    break

                case -2, Shape.X: # 太阴
                    size = (fx.r.index - self.points[-1].m.index)
                    if size >= 4:
                        # 是否满足下一状态的条件 1 → 2 → -1 → -2 → 1
                        if fx.r.low < self.points[-1].low:
                            self.__pop_bi(sys._getframe().f_lineno)
                            state = -1
                            self.log(colored("转换状态, 太阴至少阴", "red"))
                        else:
                            state = 1
                    else:
                        if fx.r.low < self.points[-1].low:
                            self.__pop_bi(sys._getframe().f_lineno)
                            state = -1
                            self.log(colored("转换状态, 太阴至少阴", "red"))
                    break

                case -2, Shape.D: # 太阴
                    size = (fx.r.index - self.points[-1].m.index)
                    if size >= 4:
                        # 是否满足下一状态的条件 1 → 2 → -1 → -2 → 1
                        if fx.m.low <= self.points[-1].m.low:
                            self.__replace_bi(fx, sys._getframe().f_lineno)
                            state = -2
                        else:
                            state = 1
                    else:
                        if fx.low <= self.points[-1].low:
                            self.__replace_bi(fx, sys._getframe().f_lineno)
                            state = -2
                    break

                case -2, Shape.S: # 太阴
                    size = fx.r.index - self.points[-1].m.index
                    if size >= 4:
                        # 是否满足下一状态的条件 1 → 2 → -1 → -2 → 1
                        state = 1
                    break

                case -2, Shape.G: # 太阴
                    if ps >= 2:
                        if fx.m is self.getHigh(self.points[-1].m, fx.r):
                            self.T = fx
                            info = "-2G"

                    size = (fx.r.index - self.points[-1].m.index)
                    relation = doubleRelation(self.points[-1].m, fx.m)
                    if size >= 4:
                        # 是否满足下一状态的条件 1 → 2 → -1 → -2 → 1
                        if fx.m.index - self.points[-1].m.index >= 4 and relation in (Direction.Up, Direction.JumpUp):
                            # 是否满足成笔的条件
                            bi = self.cklines[self.points[-1].m.index:fx.m.index+1]
                            if bi[0] != self.points[-1].m or bi[-1] != fx.m:raise Exception
                            if len(bi) < 5:raise Exception
                            if force:
                                self.__append_bi(fx, sys._getframe().f_lineno)
                                state = 2
                                break
                            high = max(bi, key=lambda x:x.high)
                            low = min(bi, key=lambda x:x.low)
                            if (low,high) == (self.points[-1].m, fx.m):
                                self.__append_bi(fx, sys._getframe().f_lineno)
                                state = 2
                        else:
                            state = 1
                    else:
                        if fx.r.low < self.points[-1].m.low:
                            self.__pop_bi(sys._getframe().f_lineno)
                            if self.points and fx.m is self.getHigh(self.points[-1].m, fx.r):
                                self.__replace_bi(fx, sys._getframe().f_lineno)
                                state = 2
                            else:
                                state = -1
                    break

                case _:
                    raise TypeError("未知分型", fx.style)
            break

        self.biMachine.state = state
        fx.m.info = info + "::" + self.state
        fx.r.info = self.state

        if self.stateStack.gettop() != state:
            self.stateStack.push(self.biMachine.state)
        self.log()

@enforce_types
class ZhongShuHandler(FeatureMachine):
    def __init__(self, level:int=1):
        super(ZhongShuHandler, self).__init__()
        self.__level = level
        self.zhshs = []
        self.zsMachine = Machine([0,1,-1])
        self.stack = Stack()
        self.logging = None

        self.features = []

        deal = self._deal_left, self._deal_mid, self._deal_right
        state = self.zsMachine.state
        features = self._left, self._mid, self._right
        self.stack.push((deal, state, features))

    @property
    def LEVEL(self):
        return self.__level
    @property
    def state(self):
        return {0: "人", 1: "天", -1: "地"}[self.zsMachine.state]

    def log(self, *args, **kwords):
        if config.debug:
            print("+", self.LEVEL, self.__class__.__name__, *args, **kwords)
            return
        if config.debug:
            config.logger.debug(self.__class__.__name__, *args, **kwords)

    @enforce_types
    def append(self, feature: [ChanCandle, ChanFeature], direction=Direction.Unknow):
        t = feature.check()
        if t:raise ChanException("ZhongShuHandler append", t)

        if self.features and (not self.features[-1].isNext(feature)):
            raise ChanException("中枢， 首尾不呼应", self.features[-1], feature)

        self.features.append(feature)

        state = self.zsMachine.state
        deal  = self._deal_left, self._deal_mid, self._deal_right

        self.log("APPEND", self.state, len(self.zhshs), deal, feature)
        while 1:
            if self.zsMachine.state == 0:
                # 准备形成中枢
                if self._deal_left:
                    self.setLeft(feature, sys._getframe().f_lineno)
                    deal = (0,1,0)

                elif self._deal_mid:
                    self.setMid(feature, sys._getframe().f_lineno)
                    deal = (0,0,1)
                    if self._left.check():
                        raise ChanException("")
                    if not self._left.isNext(feature):
                        raise Exception("不连续 1")

                elif self._deal_right:
                    self.setRight(feature, sys._getframe().f_lineno)
                    deal = (0,0,1)
                    if self._mid.check():
                        raise ChanException("")
                    if not self._mid.isNext(feature):
                        raise Exception("不连续 2")

                    zs = ChanZhongShu(feature.symbol, self._left, self._mid, self._right, feature.LEVEL)
                    if zs.ok:
                        if self._left.direction is Direction.Up:
                            state = -1 # 上下上 下跌中枢
                        elif self._left.direction is Direction.Down:
                            state = 1  # 上下上 上涨中枢
                        else:
                            raise Exception
                        self.zhshs.append(zs)
                    else:
                        self.setLeft(self._mid, sys._getframe().f_lineno)
                        self.setMid(self._right, sys._getframe().f_lineno)
                        self.setRight(None, sys._getframe().f_lineno)

                else:
                    raise Exception

            elif self.zsMachine.state in (1, -1):
                # 形成 (下上下 上涨中枢, 上下上 下跌中枢)
                zs = self.zhshs[-1]
                t = zs.add(feature)
                if t:
                    deal = (0,1,0)
                    state = 0
                    self.setLeft(feature, sys._getframe().f_lineno)

                    if self.zsMachine.state == 1:
                        name = "上涨"
                    elif self.zsMachine.state == -1:
                        name = "下跌"
                    else:
                        raise Exception(self.state)

                    ## 信号 ##
                    relation = doubleRelation(zs.interval, feature.toPillar())

                    if relation is Direction.JumpUp:
                        print("ZhongShuHandler", zs.LEVEL, f"向上突破 {name}中枢 形成 盘整三买")
                        k = min(feature.low.m.elements, key=lambda x:x.low)
                        #k.mark.update({feature.LEVEL: f"向上突破 {name}中枢 形成 盘整三买"})

                    elif relation is Direction.JumpDown:
                        print("ZhongShuHandler", zs.LEVEL, f"向下突破 {name}中枢 形成 盘整三卖")
                        k = max(feature.low.m.elements, key=lambda x:x.high)
                        #k.mark.update({feature.LEVEL: f"向下突破 {name}中枢 形成 盘整三卖"})
                    #else:
                    #    raise Exception(t, relation)

                    self.log("中枢跳出")
                    break

                if self._deal_right:
                    self.setLeft (self._mid, sys._getframe().f_lineno)
                    self.setMid(self._right, sys._getframe().f_lineno)
                    self.setRight  (feature, sys._getframe().f_lineno)
                    self.log("中枢继续")
                else:
                    raise Exception

            else:
                raise Exception

            break

        self.zsMachine.state = state
        self._deal_left, self._deal_mid, self._deal_right = deal
        self.log()

    @enforce_types
    def pop(self, tz: [ChanCandle, ChanFeature], n):
        state = self.zsMachine.state
        deal  = self._deal_left, self._deal_mid, self._deal_right

        t = None
        self.log(colored("POP", "red"), self.state, self.LEVEL, deal, tz)
        while 1:
            if self.zsMachine.state == 0:
                # 准备形成中枢
                if self._deal_left:
                    #if self.features:
                    #    t = self.features[-1]

                    self.setLeft(None, sys._getframe().f_lineno)
                    self.setMid(None, sys._getframe().f_lineno)
                    self.setRight(None, sys._getframe().f_lineno)
                    deal = (1,0,0)

                elif self._deal_mid:
                    t = self._left
                    self.setLeft(None, sys._getframe().f_lineno)
                    self.setMid(None, sys._getframe().f_lineno)
                    self.setRight(None, sys._getframe().f_lineno)
                    deal = (1,0,0)

                elif self._deal_right:
                    t = self._mid
                    #self.setLeft(None, sys._getframe().f_lineno)
                    self.setMid(None, sys._getframe().f_lineno)
                    self.setRight(None, sys._getframe().f_lineno)
                    deal = (0,1,0)
                    if self._left.check():
                        raise ChanException("")

                else:
                    raise Exception("deal", self._deal_left, self._deal_mid, self._deal_right)
                break

            elif self.zsMachine.state in (1, -1):
                # 处于中枢状态 (下上下 上涨中枢, 上下上 下跌中枢)
                zs = self.zhshs[-1]
                t = zs.pop(tz)
                if not zs.ok:
                    self.zhshs.pop()
                    if self._deal_right:
                        self.setLeft(zs.l, sys._getframe().f_lineno)
                        self.setMid(zs.m, sys._getframe().f_lineno)
                        self.setRight(None, sys._getframe().f_lineno)
                        deal = (0,0,1)
                        state = 0
                    else:
                        raise Exception
                    break

            else:
                raise Exception

            break

        self.zsMachine.state = state
        self._deal_left, self._deal_mid, self._deal_right = deal
        '''
        if not t:
            raise ValueError(self.LEVEL, "中枢处理没有数据, feature size:", len(self.features))
        if tz != t:
            raise ChanException(tz, t)
        '''
        self.features.pop()
        self.log()
        return t

@enforce_types
class FeatureHandler(FeatureMachine):
    ''' 线段处理 '''
    __slots__ = ["visual", "last", "stack", "__level", "zsh", "featureMachine", "features", "segments", "points", "lastFeature", "lastMerge", "isVisual"]
    def __init__(self, level=1, zsh:ZhongShuHandler=None):
        super(FeatureHandler, self).__init__()
        self.featureMachine = Machine([0,1,-1,2,-2])
        self.features = []
        self.segments = []
        self.points = []
        segment = None
        self.lastMerge = None
        self.logging = None

        self.__level = level
        self.zsh = zsh

        self.isVisual = False
        self.visual = None
        self.last = None

        self.stack = Stack()
        deal = self._deal_left, self._deal_mid, self._deal_right
        state = self.featureMachine.state
        candles = self._left, self._mid, self._right
        self.stack.push((deal, state, candles, 0))
        self.poped = 0
        self.appended = 0

    @enforce_types
    def getVisual(self, segment: ChanFeature):
        if self.isVisual:
            if self.featureMachine.state == 0:
                ...
            elif self.featureMachine.state == 1:
                if self._deal_left:...
                elif self._deal_left:...

            elif self.featureMachine.state == -1:
                ...
            elif self.featureMachine.state == 2:
                ...
            elif self.featureMachine.state == -2:
                ...
            else:
                ...

    @property
    def LEVEL(self):
        return self.__level
    @property
    def deal(self):
        return self._deal_left, self._deal_mid, self._deal_right
    @property
    def state(self):
        return {0:"混沌", 1:"少阳", 2:"老阳", -1:"少阴", -2:"老阴"}[self.featureMachine.state]

    def log(self, *args, **kwords):
        if config.debug:
            print("+", self.LEVEL, self.__class__.__name__, *args, **kwords)
            return
        if logging:
            config.logger.debug(self.__class__.__name__, *args, **kwords)

    def __pop_segment(self):
        point = None
        feature = None
        zfeature = None

        if self.points:
            point = self.points.pop()
            self.log(colored("pop point", "yellow"), point)

        if self.features:
            feature = self.features.pop()
            self.log(colored("pop feature", "yellow"), feature)
            self.poped += 1
            if self.zsh:
                zfeature = self.zsh.pop(feature, sys._getframe().f_lineno)

        if feature and zfeature:
            if feature != zfeature:
                print("什么情况")
            if point != feature.end:
                print("什么时候2")
        return feature

    @enforce_types
    def __append_segment(self, fx:ChanFenXing, n:int):
        zsh = self.zsh
        if fx.style in (Shape.G, Shape.D):
            if self.points:
                if self.points[-1].style == fx.style:
                    raise TypeError(self.LEVEL, "无法衔接", len(self.points), self.points[-1], fx)
            self.points.append(fx)
            self.hasUpdate = True
            self.log(colored("append point", "yellow"), fx)

            def findStart(o):
                result = None
                pos = -1
                while 1:
                    if self.segments[pos].start == o:
                        return pos
                    pos -= 1

            def findEnd(o, s):
                result = None
                pos = -1
                while 1:
                    if self.segments[pos].end == o:
                        return pos
                    if self.segments[pos].end == s:
                        raise IndexError("超出范围", pos)
                    pos -= 1

            if len(self.points) >= 2:
                start = self.points[-2]
                end = self.points[-1]
                startIndex = findStart(start)
                endIndex = findEnd(end, start)
                if endIndex == -1:
                    duan = ChanFeature(start, end, self.segments[startIndex:], self.LEVEL)
                else:
                    duan = ChanFeature(start, end, self.segments[startIndex:endIndex+1], self.LEVEL)

                if self.features:
                    if not self.features[-1].isNext(duan):
                        raise ValueError("无法连接", self.features[-1].end, duan.start, len(self.features))
                if len(self.features) >= 2:
                    if self.features[-2].end != self.features[-1].start:
                        raise ValueError("append 无法衔接")
                self.features.append(duan)
                self.log(colored("append feature", "yellow"), duan)
                self.appended += 1
                if zsh:
                    zsh.append(duan)
        else:
            raise TypeError(fx.style, "分型错误")

    @enforce_types
    def replace(self, segment:ChanFeature, n:int):
        ...

    @enforce_types
    def pop(self, n:int):
        self.log(colored("POP", "red"), self.state, "stack size:", len(self.stack.stack)-1, "segments size:", len(self.segments), "points size:", len(self.points))
        deal, state, candles, ADD = self.stack.pop()
        #self.log(self.stack.gettop())
        deal, state, candles, ADD2 = self.stack.gettop()
        self._deal_left, self._deal_mid, self._deal_right = deal
        self.featureMachine.state = state
        self._left, self._mid, self._right = candles
        flag = ADD
        while ADD:
            ADD -= 1
            self.__pop_segment()
        pop = self.segments.pop()
        self.last = None
        if self.segments:
            self.last = self.segments[-1]
        return flag, pop

    @enforce_types
    def append(self, segment:ChanFeature):
        self.poped = 0
        self.appended = 0
        if not segment:
            raise ValueError("segment is empty")
        if not segment.direction:
            raise ValueError("segment.direction is none")
        if self.segments and segment is self.segments[-1]:
            raise ValueError("feature 相同退出")
        if self.segments and not self.segments[-1].isNext(segment):
            raise ValueError("feature 首尾无法呼应", segment)

        self.isVisual = False
        ADD = 0
        self.segments.append(segment)

        state = self.featureMachine.state
        deal = self._deal_left, self._deal_mid, self._deal_right
        self.log(colored("APPEND", "green"), self.state, self.LEVEL, deal, segment)

        while 1:
            match self.featureMachine.state:
                case 0:
                    relation = segment.direction
                    if relation in (Direction.Up, Direction.JumpUp, ):
                        state = 1
                    elif relation in (Direction.Down, Direction.JumpDown, ):
                        state = -1
                    else:
                        raise TypeError(relation)
                    self.__append_segment(segment.start, sys._getframe().f_lineno)
                    ADD += 1
                    break

                case 1:
                    # 少阳
                    if segment.direction is Direction.Up:
                        self.log("不需要处理 1")
                        break

                    match (self._deal_left, self._deal_mid, self._deal_right):
                        case (1,0,0):
                            self.log(sys._getframe().f_lineno, "第一特征序列", segment)
                            deal = (0,1,0)
                            self.setLeft(segment, sys._getframe().f_lineno)

                        case (0,1,0):
                            # 第二特征序列
                            if self._left.direction != segment.direction:
                                raise Exception(self.state)

                            relation = doubleRelation(self._left.toPillar(), segment.toPillar())
                            if relation in (Direction.Up, Direction.JumpUp, Direction.Right):
                                # 涨, 逆序包含
                                self.log(sys._getframe().f_lineno, "第二特征序列", relation)
                                deal = (0,0,1)
                                self.setMid(segment, sys._getframe().f_lineno)
                                self.isVisual = True

                            elif relation in (Direction.Down, Direction.JumpDown):
                                # 跌
                                self.log(sys._getframe().f_lineno, "第二特征序列 下跌跳转", relation)
                                self.__append_segment(self._left.high, sys._getframe().f_lineno)
                                ADD += 1
                                deal = (0,1,0)
                                self.setLeft(self.segments[-2], sys._getframe().f_lineno)
                                state = -1

                            elif relation in (Direction.Left,):
                                # 顺序包含
                                self.log(sys._getframe().f_lineno, "第二特征序列 顺序包含", relation)
                                deal = (0,1,0)
                                self.lastMerge = mergeFeature(self._left, segment)
                                self.setLeft(self.lastMerge, sys._getframe().f_lineno)
                            else:
                                raise TypeError(relation)

                        case (0,0,1):
                            # 第三特征序列
                            if self._mid.direction != segment.direction:
                                raise Exception(self._mid.direction, segment.direction)

                            relation = doubleRelation(self._mid.toPillar(), segment.toPillar())
                            if relation in (Direction.Up, Direction.JumpUp, Direction.Right):
                                # 涨, 逆序包含
                                if (self._mid.direction is Direction.Down) and (self._mid.HIGH == segment.HIGH) and (relation is Direction.Right):
                                    # 特殊处理 1
                                    self.log(sys._getframe().f_lineno, "第三特征序列 特殊处理 1", relation)
                                    self.__append_segment(self._mid.high, sys._getframe().f_lineno)
                                    ADD += 1
                                    deal = (0,1,0)
                                    state = -1
                                    self.setLeft(self.segments[-2], sys._getframe().f_lineno)
                                    self.setMid(None, sys._getframe().f_lineno)
                                    break
                                self.log(sys._getframe().f_lineno, "第三特征序列", relation)
                                deal = (0,0,1)
                                self.setLeft(self._mid, sys._getframe().f_lineno)
                                self.setMid(segment, sys._getframe().f_lineno)
                                self.isVisual = True

                            elif relation in (Direction.Down, Direction.JumpDown):
                                # 下跌， 顶分型成立
                                if Direction.JumpUp is doubleRelation(self._left.toPillar(), self._mid.toPillar()):
                                    # 缺口
                                    self.log(sys._getframe().f_lineno, "第三特征序列 缺口跳转", relation)
                                    deal = (0,1,0)
                                    self.setRight(self._mid, sys._getframe().f_lineno)
                                    self.setLeft(self.segments[-2], sys._getframe().f_lineno)
                                    self.setMid(None, sys._getframe().f_lineno)
                                    state = 2
                                else:
                                    self.log(sys._getframe().f_lineno, "第三特征序列 结束跳转", relation)
                                    deal = (0,1,0)
                                    self.__append_segment(self._mid.high, sys._getframe().f_lineno)
                                    ADD += 1
                                    self.setLeft(self.segments[-2], sys._getframe().f_lineno)
                                    state = -1

                            elif relation in (Direction.Left,):
                                # 顺序包含
                                self.log(sys._getframe().f_lineno, "第三特征序列 顺序包含", relation)
                                deal = (0,0,1)
                                self.lastMerge = mergeFeature(self._mid, segment)
                                self.setMid(self.lastMerge, sys._getframe().f_lineno)
                                self.isVisual = True
                            else:
                                raise TypeError(relation)

                case -1:
                    if segment.direction is Direction.Down:
                        self.log("不需要处理 -1")
                        break

                    match (self._deal_left, self._deal_mid, self._deal_right):
                        case (1,0,0):
                            self.log(sys._getframe().f_lineno, "第一特征序列", segment)
                            deal = (0,1,0)
                            self.setLeft(segment, sys._getframe().f_lineno)

                        case (0,1,0):
                            # 第二特征序列
                            if self._left.direction != segment.direction:
                                raise Exception()

                            relation = doubleRelation(self._left.toPillar(), segment.toPillar())
                            if relation in (Direction.Up, Direction.JumpUp):
                                # 涨
                                self.log(sys._getframe().f_lineno, "第二特征序列 上涨跳转", relation)
                                self.__append_segment(self._left.low, sys._getframe().f_lineno)
                                ADD += 1
                                deal = (0,1,0)
                                self.setLeft(self.segments[-2], sys._getframe().f_lineno)
                                state = 1

                            elif relation in (Direction.Down, Direction.JumpDown, Direction.Right):
                                # 跌, 逆序包含
                                self.log(sys._getframe().f_lineno, "第二特征序列", relation)
                                deal = (0,0,1)
                                self.setMid(segment, sys._getframe().f_lineno)
                                self.isVisual = True

                            elif relation in (Direction.Left,):
                                # 顺序包含
                                self.log(sys._getframe().f_lineno, "第二特征序列 顺序包含", relation)
                                deal = (0,1,0)
                                self.lastMerge = mergeFeature(self._left, segment)
                                self.setLeft(self.lastMerge, sys._getframe().f_lineno)
                            else:
                                raise TypeError(relation)

                        case (0,0,1):
                            # 第三特征序列
                            if self._mid.direction != segment.direction:
                                raise Exception(self._mid, segment)

                            relation = doubleRelation(self._mid.toPillar(), segment.toPillar())
                            if relation in (Direction.Up, Direction.JumpUp):
                                # 涨
                                if Direction.JumpDown is doubleRelation(self._left.toPillar(), self._mid.toPillar()):
                                    # 缺口
                                    self.log(sys._getframe().f_lineno, "第三特征序列 缺口 跳转", relation)
                                    deal = (0,1,0)
                                    self.setRight(self._mid, sys._getframe().f_lineno)
                                    self.setLeft(self.segments[-2], sys._getframe().f_lineno)
                                    self.setMid(None, sys._getframe().f_lineno)
                                    state = -2
                                else:
                                    self.log(sys._getframe().f_lineno, "第三特征序列 结束 跳转", relation)
                                    self.__append_segment(self._mid.low, sys._getframe().f_lineno)
                                    ADD += 1
                                    deal = (0,1,0)
                                    self.setLeft(self.segments[-2], sys._getframe().f_lineno)
                                    # self.setMid(None, sys._getframe().f_lineno)
                                    state = 1

                            elif relation in (Direction.Down, Direction.JumpDown, Direction.Right):
                                # 跌, 逆序包含
                                if (self._mid.direction is Direction.Up) and (self._mid.LOW == segment.LOW) and (relation is Direction.Right):
                                    # 特殊处理 1
                                    self.log(sys._getframe().f_lineno, "第三特征序列 特殊处理 1", relation)
                                    self.__append_segment(self._mid.low, sys._getframe().f_lineno)
                                    ADD += 1
                                    deal = (0,1,0)
                                    state = 1
                                    self.setLeft(self.segments[-2], sys._getframe().f_lineno)
                                    self.setMid(None, sys._getframe().f_lineno)
                                    break

                                self.log(sys._getframe().f_lineno, "第三特征序列", relation)
                                deal = (0,0,1)
                                self.setLeft(self._mid, sys._getframe().f_lineno)
                                self.setMid(segment, sys._getframe().f_lineno)
                                self.isVisual = True

                            elif relation in (Direction.Left,):
                                # 顺序包含
                                self.log(sys._getframe().f_lineno, "第三特征序列 顺序包含", relation)
                                deal = (0,0,1)
                                self.lastMerge = mergeFeature(self._mid, segment)
                                self.setMid(self.lastMerge, sys._getframe().f_lineno)
                                self.isVisual = True

                            else:
                                raise TypeError(relation)

                case 2:
                    if segment.direction is Direction.Down:
                        self.log("不需要处理 2")
                        break
                    # 进入缺口模式
                    match (self._deal_left, self._deal_mid, self._deal_right):
                        case (0,1,0):
                            # 第二特征序列
                            self.isVisual = True
                            if self._left.direction != segment.direction:
                                raise Exception()

                            relation = doubleRelation(self._left.toPillar(), segment.toPillar())
                            if relation in (Direction.Up, Direction.JumpUp, Direction.Right, Direction.Down, Direction.JumpDown):
                                # 涨, 跌, 逆序包含
                                self.log(sys._getframe().f_lineno, "第二特征序列 缺口模式", relation)
                                deal = (0,0,1)
                                self.setMid(segment, sys._getframe().f_lineno)

                            elif relation in (Direction.Left,):
                                # 顺序包含
                                self.log(sys._getframe().f_lineno, "第二特征序列 缺口模式 顺序包含", relation)
                                deal = (0,1,0)
                                self.lastMerge = mergeFeature(self._left, segment)
                                self.setLeft(self.lastMerge, sys._getframe().f_lineno)

                            else:
                                raise TypeError(relation)

                        case (0,0,1):
                            # 第三特征序列
                            if self._mid.direction != segment.direction:
                                raise Exception("应该:", str(segment.direction), "实际:", str(self._mid.direction))
                            relation = doubleRelation(self._mid.toPillar(), segment.toPillar())
                            if relation in (Direction.Up, Direction.JumpUp):
                                # 涨
                                self.log(sys._getframe().f_lineno, "第三特征序列 缺口模式 跳转", relation)
                                self.__append_segment(self._right.high, sys._getframe().f_lineno)
                                self.__append_segment(self._mid.low, sys._getframe().f_lineno)
                                ADD += 2
                                deal = (0,1,0)
                                self.setLeft(self.segments[-2], sys._getframe().f_lineno)#
                                if self._left.direction != Direction.Down:
                                    raise Exception()
                                state = 1

                            elif relation in (Direction.Down, Direction.JumpDown):
                                # 下跌
                                self.log(sys._getframe().f_lineno, "第三特征序列 缺口模式 跳转", relation)
                                self.__append_segment(self._right.high, sys._getframe().f_lineno)
                                ADD += 1
                                deal = (0,0,1)
                                self.setLeft(self.segments[-3], sys._getframe().f_lineno)#
                                self.setMid(self.segments[-1], sys._getframe().f_lineno)#
                                if self._left.direction != self._mid.direction or self._mid.direction is Direction.Down:
                                    raise Exception()
                                state = -1

                            elif relation in (Direction.Left,):
                                # 顺序包含
                                self.log(sys._getframe().f_lineno, "第三特征序列 缺口模式 顺序包含", relation)
                                deal = (0,0,1)
                                self.lastMerge = mergeFeature(self._mid, segment)
                                self.setMid(self.lastMerge, sys._getframe().f_lineno)
                                self.isVisual = True
                            elif relation in (Direction.Right,):
                                # 逆序包含
                                self.log(sys._getframe().f_lineno, "第三特征序列 缺口模式 逆序包含", relation)
                                deal = (0,0,1)
                                self.setLeft(self._mid, sys._getframe().f_lineno)
                                self.setMid(segment, sys._getframe().f_lineno)
                                self.isVisual = True
                            else:
                                raise TypeError(relation)

                case -2:
                    if segment.direction is Direction.Up:
                        self.log("不需要处理 -2")
                        break
                    # 下跌出现底
                    # 进入缺口模式
                    match (self._deal_left, self._deal_mid, self._deal_right):
                        case (0,1,0):
                            # 第二特征序列
                            self.isVisual = True
                            if self._left.direction != segment.direction:
                                raise Exception()
                            relation = doubleRelation(self._left.toPillar(), segment.toPillar())
                            if relation in (Direction.Up, Direction.JumpUp, Direction.Right, Direction.Down, Direction.JumpDown):
                                # 涨, 跌, 逆序包含
                                self.log(sys._getframe().f_lineno, "第二特征序列 缺口模式", relation)
                                deal = (0,0,1)
                                self.setMid(segment, sys._getframe().f_lineno)

                            elif relation in (Direction.Left,):
                                # 顺序包含
                                self.log(sys._getframe().f_lineno, "第二特征序列 缺口模式 顺序包含", relation)
                                deal = (0,1,0)
                                self.lastMerge = mergeFeature(self._left, segment)
                                self.setLeft(self.lastMerge, sys._getframe().f_lineno)
                            else:
                                raise TypeError(relation)
                            self.isVisual = True

                        case (0,0,1):
                            # 第三特征序列
                            if self._mid.direction != segment.direction:
                                raise Exception("应该:", str(segment.direction), "实际:", str(self._mid.direction))

                            relation = doubleRelation(self._mid.toPillar(), segment.toPillar())
                            if relation in (Direction.Up, Direction.JumpUp):
                                # 涨
                                self.log(sys._getframe().f_lineno, "第三特征序列 缺口模式 正常跳转", relation)
                                self.__append_segment(self._right.low, sys._getframe().f_lineno)
                                ADD += 1
                                deal = (0,0,1)
                                self.setLeft(self.segments[-3], sys._getframe().f_lineno)
                                self.setMid(self.segments[-1], sys._getframe().f_lineno)
                                state = 1

                            elif relation in (Direction.Down, Direction.JumpDown):
                                # 跌
                                self.log(sys._getframe().f_lineno, "第三特征序列 缺口模式 正常跳转", relation)
                                self.__append_segment(self._right.low, sys._getframe().f_lineno)
                                self.__append_segment(self._mid.high, sys._getframe().f_lineno)
                                ADD += 2
                                deal = (0,1,0)
                                self.setLeft(self.segments[-2], sys._getframe().f_lineno)
                                state = -1

                            elif relation in (Direction.Left,):
                                # 顺序包含
                                self.log(sys._getframe().f_lineno, "第三特征序列 缺口模式 顺序包含", relation)
                                deal = (0,0,1)
                                self.lastMerge = mergeFeature(self._mid, segment)
                                self.setMid(self.lastMerge, sys._getframe().f_lineno)
                                self.isVisual = True

                            elif relation in (Direction.Right,):
                                # 逆序包含
                                self.log(sys._getframe().f_lineno, "第三特征序列 缺口模式 逆序包含", relation)
                                deal = (0,0,1)
                                self.setLeft(self._mid, sys._getframe().f_lineno)
                                self.setMid(segment, sys._getframe().f_lineno)
                                self.isVisual = True
                            else:
                                raise TypeError(relation)
            break

        self.last = segment
        self.featureMachine.state = state
        self._deal_left, self._deal_mid, self._deal_right = deal
        candles = self._left, self._mid, self._right
        self.stack.push((deal, state, candles, ADD))
        self.log()

class ChanZouShi:
    ...

class ChanQuShi:
    ...

class ChanPanZheng:
    ...

class ChanShangZhang:
    ...

class ChanXiaDie:
    ...

class ZouShiHandler:
    ...

class ChZhShCh:
    def __init__(self):
        #super(BiHandler, self).__init__(0, zsh)
        self.cklines = [] # 缠论k线
        self.klines = []

        self.klineHandler = KlineHandler(self.klines, self.cklines)
        self.shapeHandler = FenXingHandler()

        self.biHandler = BiHandler(self.cklines)
        self.bzsHandler = ZhongShuHandler(self.biHandler.LEVEL)
        self.biHandler.zsh = self.bzsHandler
        #self.biHandler.debug = 1

        self.featureHandler = FeatureHandler(1)
        self.zsHandler = ZhongShuHandler(self.featureHandler.LEVEL)
        self.featureHandler.zsh = self.zsHandler
        #self.featureHandler = None

    def get_klines_by_datetime(self, start:datetime, end:datetime):
        result = []
        flag = 0
        for k in self.klines:
            if k.dt == start:flag=1
            if flag:result.append(k)
            if k.dt == end:break
        return result

    def getDebugInfo(self):
        """ 将发现的问题发送邮件
        """
        from io import BytesIO
        bio = BytesIO()
        for k in self.klines:
            bio.write(bytes(k))
        with open(f"chzhshch_log_{int(datetime.now().timestamp())}", "wb") as f:
            f.write(bio.getbuffer())
            print("文件保存在:", f.name)

    def __getitem__(self, obj):
        return self.cklines[obj]

    def __len__(self):
        return len(self.cklines)

    def pop(self):
        hasReplaced, ck = self.klineHandler.pop()
        self.shapeHandler.pop()
        fx = None
        if hasReplaced:
            fx = self.shapeHandler.append(ck)
            self.biHandler.append(fx)
            if self.featureHandler:
                state = (self.biHandler.appended, self.biHandler.poped)
                if state == (0,0):
                    ...
                elif state == (1,0):
                    if not self.biHandler.hasUpdated:raise Exception
                    self.featureHandler.append(self.biHandler.features[-1])

                elif state == (0,1):
                    if not self.biHandler.hasBacked:raise Exception
                    self.featureHandler.pop(sys._getframe().f_lineno)

                elif state == (1,1):
                    #if not self.biHandler.hasReplaced:raise Exception
                    self.featureHandler.pop(sys._getframe().f_lineno)
                    self.featureHandler.append(self.biHandler.features[-1])
                elif state == (1,2):
                    self.featureHandler.pop(sys._getframe().f_lineno)
                    self.featureHandler.pop(sys._getframe().f_lineno)
                    self.featureHandler.append(self.biHandler.features[-1])
                else:
                    raise ValueError("未处理此状态", state)
        else:
            self.biHandler.pop()
            if self.featureHandler:
                state = (self.biHandler.appended, self.biHandler.poped)
                if state == (0,0):
                    ...
                elif state == (0,1):
                    if not self.biHandler.hasBacked:raise Exception
                    self.featureHandler.pop(sys._getframe().f_lineno)
                else:
                    raise ValueError("未处理此状态", state)

    @enforce_types
    def add(self, k: RawCandle):
        # 当下模式
        hasReplaced, ck = self.klineHandler.append(k)
        #print(k, ck)
        #print("#"*55)
        if len(self.cklines) >= 3:
            fx = ChanFenXing(*self.cklines[-3:])#self.shapeHandler.append(ck)
            if fx != None:
                self.biHandler.append(fx)
            if self.featureHandler:
                state = (self.biHandler.appended, self.biHandler.poped)
                if state == (0,0):
                    ...
                elif state == (1,0):
                    if not self.biHandler.hasUpdated:raise Exception
                    self.featureHandler.append(self.biHandler.features[-1])

                elif state == (0,1):
                    if not self.biHandler.hasBacked:raise Exception
                    self.featureHandler.pop(sys._getframe().f_lineno)

                elif state == (1,1):
                    #if not self.biHandler.hasReplaced:raise Exception
                    self.featureHandler.pop(sys._getframe().f_lineno)
                    self.featureHandler.append(self.biHandler.features[-1])
                elif state == (1,2):
                    self.featureHandler.pop(sys._getframe().f_lineno)
                    self.featureHandler.pop(sys._getframe().f_lineno)
                    self.featureHandler.append(self.biHandler.features[-1])
                elif state == (2,2):
                    self.featureHandler.pop(sys._getframe().f_lineno)
                    self.featureHandler.append(self.biHandler.features[-1])
                else:
                    raise ValueError("未处理此状态", state)

    @enforce_types
    def add2(self, k: RawCandle):
        # 当下模式
        hasReplaced, ck = self.klineHandler.append(k)
        #print(k, ck)
        #print("#"*55)
        if hasReplaced:
            backed = self.shapeHandler.pop()
            #if backed:print("回退")
        fx = self.shapeHandler.append(ck)
        if fx != None:
            self.biHandler.append(fx)
            #return
            if self.featureHandler:
                state = (self.biHandler.appended, self.biHandler.poped)
                if state == (0,0):
                    ...
                elif state == (1,0):
                    if not self.biHandler.hasUpdated:raise Exception
                    self.featureHandler.append(self.biHandler.features[-1])

                elif state == (0,1):
                    if not self.biHandler.hasBacked:raise Exception
                    self.featureHandler.pop(sys._getframe().f_lineno)

                elif state == (1,1):
                    #if not self.biHandler.hasReplaced:raise Exception
                    self.featureHandler.pop(sys._getframe().f_lineno)
                    self.featureHandler.append(self.biHandler.features[-1])
                elif state == (1,2):
                    self.featureHandler.pop(sys._getframe().f_lineno)
                    self.featureHandler.pop(sys._getframe().f_lineno)
                    self.featureHandler.append(self.biHandler.features[-1])
                elif state == (2,2):
                    self.featureHandler.pop(sys._getframe().f_lineno)
                    self.featureHandler.append(self.biHandler.features[-1])
                else:
                    raise ValueError("未处理此状态", state)

    @enforce_types
    def addraw(self, symbol:str, dt:datetime, open:[int, float], high:[int, float], low:[int, float], close:[int, float], vol:[int, float]):
        k = RawCandle(symbol, dt, open, high, low, close, vol)
        self.add(k)

    @enforce_types
    def toCharts(self, path:str= "lines.html", useReal=False):
        '''if self.featureHandler:
            for feature in self.biHandler.features:
                self.featureHandler.append(feature)
        '''
        import echarts_plot # czsc
        reload(echarts_plot)
        #from czsc.util.echarts_plot import kline_pro
        kline_pro = echarts_plot.kline_pro
        bi = []
        for ck in self.biHandler.points:
            k = ck
            if k.style == Shape.G:
                t = max(k.elements, key=lambda x:x.high)
                if useReal:
                    bi.append({"dt":t.dt, "bi": t.high})
                else:
                    bi.append({"dt":k.dt, "bi": t.high})
            elif k.style == Shape.D:
                t = min(k.elements, key=lambda x:x.low)
                if useReal:
                    bi.append({"dt":t.dt, "bi": t.low})
                else:
                    bi.append({"dt":k.dt, "bi": t.low})
            else:
                raise ChanException("unknow shape, tv")
                continue
        result = []
        for kline in self.cklines:
            tmp = dict()
            x = ""
            if kline.info:# path://M512 938.666667L262.4 640h499.2z M426.666667 128h170.666666v576h-170.666666z
                x = kline.dt
                tmp.update({"name": kline.info, "value":kline.info, "coord": [x, (kline.low+kline.high)/2], "itemStyle":{'color': 'rgb(255,0,0)'}, "symbol": "path://M512 938.666667L262.4 640h499.2z M426.666667 128h170.666666v576h-170.666666z"})
            tmp.update({"symbolSize":7})
            result.append(tmp)
        mark = result

        #bi = [{"dt":k.dt, "bi": k.low if k.style is Shape.D else k.high} for k in self.biHandler.points]
        xd = []
        if self.featureHandler:
            size = len(self.featureHandler.points)

            for i in range(size):
                k = self.featureHandler.points[i]

                if k.style is Shape.G:
                    if useReal:
                        k = max(k.elements, key=lambda x:x.high)
                    point = k.high
                if k.style is Shape.D:
                    if useReal:
                        k = min(k.elements, key=lambda x:x.low)
                    point = k.low
                xd.append({
                  'dt': k.dt,
                  'xd': point,
                  'bi': point})

        if useReal:
            charts = kline_pro([x.candleDict() for x in self.klines], bi = bi, xd=xd, mark=mark,title=self.klines[0].symbol)
        else:
            charts = kline_pro([x.candleDict() for x in self.cklines], bi = bi, xd=xd, mark=mark,title=self.cklines[0].symbol)

        charts.render(path)
        return charts

class Bitstamp:
    def __init__(self, pair, m, size):
        self.czsc = ChZhShCh()
        self.freq = m * 60
        self.pair = pair
        NetKLine.getBitstamp(self.czsc, pair, self.freq, size)

    def update(self):
        js = Bitstamp.get(self.pair, self.freq, self.czsc.klines[-1].dt, None)
        for item in js:
            open = item["open"]
            high = item["high"]
            low  = item["low"]
            close = item["close"]
            volume = item["volume"]
            timestamp = item["timestamp"]
            dt = datetime.fromtimestamp(int(timestamp))
            kline = RawCandle(self.pair, dt, float(open), float(high), float(low), float(close), float(volume))
            self.czsc.add(kline)

    @classmethod
    def get(cls, pair:str, step:int, start:datetime=datetime.now(), end:datetime=datetime.now()):
        pairs = "btcusd, btceur, btcgbp, btcpax, btcusdt, btcusdc, gbpusd, gbpeur, eurusd, ethusd, etheur, ethbtc, ethgbp, ethpax, ethusdt, ethusdc, xrpusd, xrpeur, xrpbtc, xrpgbp, xrppax, xrpusdt, uniusd, unieur, unibtc, ltcusd, ltceur, ltcbtc, ltcgbp, linkusd, linkeur, linkbtc, linkgbp, linketh, xlmusd, xlmeur, xlmbtc, xlmgbp, bchusd, bcheur, bchbtc, bchgbp, aaveusd, aaveeur, aavebtc, algousd, algoeur, algobtc, compusd, compeur, compbtc, snxusd, snxeur, snxbtc, batusd, bateur, batbtc, mkrusd, mkreur, mkrbtc, zrxusd, zrxeur, zrxbtc, yfiusd, yfieur, yfibtc, grtusd, grteur, umausd, umaeur, umabtc, omgusd, omgeur, omgbtc, omggbp, kncusd, knceur, kncbtc, crvusd, crveur, crvbtc, audiousd, audioeur, audiobtc, usdtusd, usdteur, usdcusd, usdceur, usdcusdt, daiusd, paxusd, paxeur, paxgbp, eth2eth, gusdusd".split(", ")
        if not pair in pairs:
            raise ValueError("没有此交易对", pair)
        steps = [60, 180, 300, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400, 259200]
        headers = {"Connection":"keep-alive", "Upgrade-Insecure-Requests":"1", "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8", "Accept-Encoding":"gzip, deflate", "Accept-Language":"zh-CN,en-US;q=0.9", "X-Requested-With":"mark.via"}
        s = requests.Session()
        s.headers=headers
        start = int(start.timestamp())
        MAX = int(datetime(2013,7,1).timestamp())
        if start < MAX:
            start = MAX
            print("超出范围")
            return []

        result = []
        startStamp = start#time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.localtime(start))
        times = [startStamp, ]
        count = 0
        while 1:
            url = f"https://www.bitstamp.net/api/v2/ohlc/{pair}/?step={step}&limit=1000&start={startStamp}"
            resp = s.get(url, timeout=5)
            dat = resp.json()
            #print(dat)
            result.extend(dat["data"]["ohlc"])
            if len(dat["data"]["ohlc"]) < 1000:
                print("exit")
                break
            startStamp = int(dat["data"]["ohlc"][-1]["timestamp"])
            if startStamp in times:
                print("重复获取")
            count += 1
        print("获取数量", count)
        return result

class NetKLine:
    @classmethod
    def TS(cls, handler, code, start, end, freq):
        df = ts.pro_bar(ts_code=code, asset='I', start_date=start, end_date=end, freq=freq)
        df = df.sort_index(ascending=False)

        return df

    @classmethod
    @enforce_types
    def getBitstamp2(cls, handler:ChZhShCh = None, pair:str="btcusd", step:int=60, start:datetime=datetime.now(), end:datetime=datetime.now()):
        pairs = "btcusd, btceur, btcgbp, btcpax, btcusdt, btcusdc, gbpusd, gbpeur, eurusd, ethusd, etheur, ethbtc, ethgbp, ethpax, ethusdt, ethusdc, xrpusd, xrpeur, xrpbtc, xrpgbp, xrppax, xrpusdt, uniusd, unieur, unibtc, ltcusd, ltceur, ltcbtc, ltcgbp, linkusd, linkeur, linkbtc, linkgbp, linketh, xlmusd, xlmeur, xlmbtc, xlmgbp, bchusd, bcheur, bchbtc, bchgbp, aaveusd, aaveeur, aavebtc, algousd, algoeur, algobtc, compusd, compeur, compbtc, snxusd, snxeur, snxbtc, batusd, bateur, batbtc, mkrusd, mkreur, mkrbtc, zrxusd, zrxeur, zrxbtc, yfiusd, yfieur, yfibtc, grtusd, grteur, umausd, umaeur, umabtc, omgusd, omgeur, omgbtc, omggbp, kncusd, knceur, kncbtc, crvusd, crveur, crvbtc, audiousd, audioeur, audiobtc, usdtusd, usdteur, usdcusd, usdceur, usdcusdt, daiusd, paxusd, paxeur, paxgbp, eth2eth, gusdusd".split(", ")
        if not pair in pairs:
            raise ValueError("没有此交易对", pair)
        steps = [60, 180, 300, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400, 259200]
        headers = {"Connection":"keep-alive", "Upgrade-Insecure-Requests":"1", "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8", "Accept-Encoding":"gzip, deflate", "Accept-Language":"zh-CN,en-US;q=0.9", "X-Requested-With":"mark.via"}
        s = requests.Session()
        s.headers=headers
        start = int(start.timestamp())
        MAX = int(datetime(2013,7,1).timestamp())
        if start < MAX:
            start = MAX
            print("超出范围")

        js = []
        startStamp = start#time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.localtime(start))
        times = [startStamp, ]
        count = 0
        while 1:
            url = f"https://www.bitstamp.net/api/v2/ohlc/{pair}/?step={step}&limit=1000&start={startStamp}"
            resp = s.get(url, timeout=5)
            dat = resp.json()
            #print(dat)
            js.extend(dat["data"]["ohlc"])
            print(url)
            if handler != None:
                for item in dat["data"]["ohlc"]:
                    open = item["open"]
                    high = item["high"]
                    low  = item["low"]
                    close = item["close"]
                    volume = item["volume"]
                    timestamp = item["timestamp"]
                    dt = datetime.fromtimestamp(int(timestamp))
                    if dt > end:break
                    if dt in times:continue
                    kline = RawCandle(pair, dt, float(open), float(high), float(low), float(close), float(volume))
                    handler.add(kline)
                    kline.index = count
                    count += 1
                    times.append(dt)
                    #print(dt)
                    #print(count)

            if len(dat["data"]["ohlc"]) < 1000:
                print("exit")
                break
            startStamp = int(dat["data"]["ohlc"][-1]["timestamp"])
            if startStamp in times:
                print("重复获取")
        print("获取数量", count)
        return handler, js

    @classmethod
    @enforce_types
    def getBitstamp(cls, handler:ChZhShCh = None, pair:str="btcusd", step:int=60, limit:int=600):
        pairs = "btcusd, btceur, btcgbp, btcpax, btcusdt, btcusdc, gbpusd, gbpeur, eurusd, ethusd, etheur, ethbtc, ethgbp, ethpax, ethusdt, ethusdc, xrpusd, xrpeur, xrpbtc, xrpgbp, xrppax, xrpusdt, uniusd, unieur, unibtc, ltcusd, ltceur, ltcbtc, ltcgbp, linkusd, linkeur, linkbtc, linkgbp, linketh, xlmusd, xlmeur, xlmbtc, xlmgbp, bchusd, bcheur, bchbtc, bchgbp, aaveusd, aaveeur, aavebtc, algousd, algoeur, algobtc, compusd, compeur, compbtc, snxusd, snxeur, snxbtc, batusd, bateur, batbtc, mkrusd, mkreur, mkrbtc, zrxusd, zrxeur, zrxbtc, yfiusd, yfieur, yfibtc, grtusd, grteur, umausd, umaeur, umabtc, omgusd, omgeur, omgbtc, omggbp, kncusd, knceur, kncbtc, crvusd, crveur, crvbtc, audiousd, audioeur, audiobtc, usdtusd, usdteur, usdcusd, usdceur, usdcusdt, daiusd, paxusd, paxeur, paxgbp, eth2eth, gusdusd".split(", ")
        if not pair in pairs:
            raise ValueError("没有此交易对", pair)
        steps = [60, 180, 300, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400, 259200]
        headers = {"Connection":"keep-alive", "Upgrade-Insecure-Requests":"1", "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8", "Accept-Encoding":"gzip, deflate", "Accept-Language":"zh-CN,en-US;q=0.9", "X-Requested-With":"mark.via"}
        s = requests.Session()
        s.headers=headers
        start = int(datetime.now().timestamp()) - (limit*step)# - (8*60*60) +step
        MAX = int(datetime(2013,7,1).timestamp())
        if start < MAX:
            start = MAX
            print("超出范围")

        js = []
        startStamp = start#time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.localtime(start))
        times = [startStamp, ]
        count = 0
        while 1:
            url = f"https://www.bitstamp.net/api/v2/ohlc/{pair}/?step={step}&limit=1000&start={startStamp}"
            resp = s.get(url, timeout=5)
            dat = resp.json()
            #print(dat)
            js.extend(dat["data"]["ohlc"])
            print(url)
            if handler != None:
                for item in dat["data"]["ohlc"]:
                    open = item["open"]
                    high = item["high"]
                    low  = item["low"]
                    close = item["close"]
                    volume = item["volume"]
                    timestamp = item["timestamp"]
                    dt = datetime.fromtimestamp(int(timestamp))
                    if dt in times:continue
                    kline = RawCandle(pair, dt, float(open), float(high), float(low), float(close), float(volume))
                    handler.add(kline)
                    kline.index = count
                    count += 1
                    times.append(dt)
                    #print(dt)
                    #print(count)

            if len(dat["data"]["ohlc"]) < 1000:
                print("exit")
                break
            startStamp = int(dat["data"]["ohlc"][-1]["timestamp"])
            if startStamp in times:
                print("重复获取")
        print("获取数量", count)
        return handler, js

    @classmethod
    @enforce_types
    def getOKEX(cls, handler:ChZhShCh = None, pair:str="BTC-USDT-SWAP", step:int=60, limit:int=600, style:str="swap"):
        steps = [60, 180, 300, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400, 604800, 2678400, 8035200, 16070400, 31536000]
        headers = {"Connection":"keep-alive", "Upgrade-Insecure-Requests":"1", "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8", "Accept-Encoding":"gzip, deflate", "Accept-Language":"zh-CN,en-US;q=0.9", "X-Requested-With":"mark.via"}
        s = requests.Session()
        s.headers=headers
        start = int(datetime.now().timestamp()) - (limit*step) - (8*60*60)

        js = []
        startStamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.localtime(start))
        times = [startStamp, ]
        count = 0
        while True:
            #time.sleep(0.25)
            url = f"https://www.okex.com/api/{style}/v3/instruments/{pair}/candles?granularity={step}&start={startStamp}"
            resp = s.get(url, timeout=5)
            dat = resp.json()
            print(url)
            print(len(dat), dat[0][0], dat[-1][0])
            js.extend(dat)
            #print(url)
            #print(dat)
            if handler != None:
                if "code" in dat:
                    print(dat)
                    raise ValueError(dat)

                array = dat[::-1]

                for item in array:
                    try:
                        timestamp, open, high, low, close, volume = item
                    except:
                        # 合约
                        timestamp, open, high, low, close, volume, vol = item

                    dt = datetime.fromtimestamp(28800+int(time.mktime(time.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.000Z"))))
                    if dt in times:continue
                    kline = RawCandle(pair, dt, float(open), float(high), float(low), float(close), float(volume))
                    handler.add(kline)
                    kline.index = count
                    count += 1
                    times.append(dt)
                    #print(dt)
            if len(dat) < 200:break
            startStamp = dat[0][0]
            if startStamp in times:
                print("重复获取")
        print(count)
        return handler, js

def test(points=[3,2,5,3,7,4,7,2.5,5,4,8,6]):
    # 测试
    candles = BiHandler.generator(points)
    obj = ChZhShCh()
    for p in candles:
        obj.add(p)
    obj.toCharts()
    return obj

@timeit
def test2():
    with open("ltc1633785266", "rb") as f:
        dat = f.read()
        czsc = ChZhShCh()
        i = 1
        while dat:
            k = RawCandle.frombytes(dat[:48], "ltcusd")
            #if k.dt > datetime(2021,10,8,20,10):break
            dat = dat[48:]
            try:czsc.add(k)
            except:
                print("异常退出")
                traceback.print_exc()
                break
            i += 1
        print(i)
        czsc.toCharts()

if __name__ == '__main__':
    test()

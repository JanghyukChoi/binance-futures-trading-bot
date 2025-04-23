import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from binance import Client
import pandas as pd
import numpy as np
import concurrent.futures
import logging
from config_template import api_key, api_secret


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("autotrading.log", mode="a"),
        logging.StreamHandler()
    ]
)

client = Client()

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
    }
})


binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
    },
})

##################################################
# 1) OHLCV 및 지표 계산
##################################################
def fetch_ohlcv_data(symbol, timeframe='1h', limit=140):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # 숫자형 변환
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].apply(pd.to_numeric, errors='coerce')

    # fillna
    df['high'] = df['high'].ffill()
    df['low']  = df['low'].ffill()
    return df

def calculate_consecutive(cond_series: pd.Series) -> pd.Series:
    arr = cond_series.values
    result = []
    count = 0
    for i in range(len(arr)):
        if arr[i]:
            if i > 0 and arr[i-1]:
                count += 1
            else:
                count = 1
        else:
            count = 0
        result.append(count)
    return pd.Series(result, index=cond_series.index)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """주요 지표 계산 + 윌리엄스 프랙탈(5) 저점(단순 종가) 계산까지 포함"""
    # 1) 이동평균, 볼린저, 일목 등
    df['sma10']  = df['close'].rolling(10).mean()
    df['sma20']  = df['close'].rolling(20).mean()
    df['sma120'] = df['close'].rolling(120).mean()

    df['highest_120'] = df['close'].rolling(120).max()
    df['lowest_120']  = df['close'].rolling(120).min()

    std = df['close'].rolling(20).std()
    df['upper_band'] = df['sma20'] + 2*std
    df['lower_band'] = df['sma20'] - 2*std
    df['bbw'] = ((df['upper_band'] - df['lower_band']) / df['sma20']) * 100

    df['sma_volume20'] = df['volume'].rolling(20).mean()

    df['candle_body'] = (df['close'] - df['open']).abs()
    df['avg_candle_body_20'] = df['candle_body'].shift(1).rolling(20).mean()

    df['green_candle'] = df['close'] > df['open']
    df['red_candle']   = df['close'] < df['open']
    df['consecutive_greens'] = calculate_consecutive(df['green_candle'])
    df['consecutive_reds']   = calculate_consecutive(df['red_candle'])

    df['sma10_up'] = df['sma10'].diff() > 0
    df['sma10_down'] = df['sma10'].diff() < 0
    df['consecutive_sma10_up']   = calculate_consecutive(df['sma10_up'])
    df['consecutive_sma10_down'] = calculate_consecutive(df['sma10_down'])

    df['sma120_up'] = df['sma120'].diff() > 0
    df['sma120_down'] = df['sma120'].diff() < 0
    df['consecutive_sma120_up']   = calculate_consecutive(df['sma120_up'])
    df['consecutive_sma120_down'] = calculate_consecutive(df['sma120_down'])

    df['sma20_up']   = df['sma20'].diff() > 0
    df['sma20_down'] = df['sma20'].diff() < 0
    df['consecutive_sma20_up']   = calculate_consecutive(df['sma20_up'])
    df['consecutive_sma20_down'] = calculate_consecutive(df['sma20_down'])

    # 일목균형
    conv_line = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    base_line = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['span1'] = ((conv_line + base_line)/2).shift(26)
    df['span2'] = (df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2
    df['span2'] = df['span2'].shift(26)

    df['sma10_gt_sma20'] = df['sma10'] > df['sma20']
    df['consecutive_sma10_gt_sma20'] = calculate_consecutive(df['sma10_gt_sma20'])

    df['sma10_lt_sma20'] = df['sma10'] < df['sma20']
    df['consecutive_sma10_lt_sma20'] = calculate_consecutive(df['sma10_lt_sma20'])

    df['sma5'] = df['close'].rolling(5).mean()
    df['sma5_up'] = df['sma5'].diff() > 0
    df['sma5_down'] = df['sma5'].diff() < 0
    df['bbw_down'] = df['bbw'].diff() < 0

    # 2) 윌리엄스 프랙탈(5) 저점: close 기준
    #    df['fractal_low5_value'] 에 기록 (NaN or 값)
    df['fractal_low5_value'] = np.nan
    closes = df['close'].values
    idx_array = df.index.to_list()  # 실제 인덱스 리스트

    for i in range(2, len(df) - 2):
        c0 = closes[i]
        if (c0 < closes[i-1] and c0 < closes[i-2] 
                and c0 < closes[i+1] and c0 < closes[i+2]):
            # .at[...] 사용 권장 -> SettingWithCopyWarning 피하기
            df.at[idx_array[i], 'fractal_low5_value'] = c0

    return df


##################################################
# 2) 기타 보조함수
##################################################
def is_close_above_70_high(df):
    if len(df) < 71:
        return False
    highest_70 = df['close'].iloc[-71:-1].max()
    last_close = df['close'].iloc[-1]
    if highest_70 <= 0:
        return False
    diff_pct = (last_close - highest_70) / highest_70 * 100
    return (diff_pct >= 1.0)

def is_close_below_70_low(df):
    if len(df) < 71:
        return False
    lowest_70 = df['close'].iloc[-71:-1].min()
    last_close = df['close'].iloc[-1]
    if lowest_70 <= 0:
        return False
    diff_pct = (lowest_70 - last_close) / lowest_70 * 100
    return (diff_pct >= 1.0)

##################################################
# 3) 업트렌드 / 다운트렌드 체크
##################################################
def check_uptrend(df: pd.DataFrame, sym) -> bool:
    """프랙탈(5) 포함 속도 최적화 업트렌드 판별"""
    if len(df) < 120:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]

    last_close = last['close']
    last_open  = last['open']
    current_body_return = abs((last_close - last_open) / last_open) * 100

    prev_close = prev['close']
    prev_open  = prev['open']
    previous_body_return = abs((prev_close - prev_open) / prev_open) * 100

    # 최근 15~20봉 전처리
    last_20_bbw = df['bbw'].iloc[-20:]
    last_15_bbw_min = last_20_bbw.tail(15).min()
    recent_10_green = df['green_candle'].iloc[-10:].sum()

    # 1) 몸통 20평 10배 초과 -> 제외
    if last['candle_body'] > 10 * last['avg_candle_body_20']:
        return False

    # 2) 메인 업트렌드 조건
    uptrend = (
        last_close >= last['highest_120'] and
        ((last_close - last['lowest_120']) / last['lowest_120']) <= 2 and
        last['volume'] > last['sma_volume20'] and
        last['consecutive_greens'] < 8 and
        last['sma10'] > last['sma20'] and
        prev['sma10'] > prev['sma20'] and
        last_close > last['upper_band'] and
        last['upper_band'] > prev['upper_band'] and
        last['lower_band'] < prev['lower_band'] and
        last['consecutive_sma10_gt_sma20'] < 20 and
        last['consecutive_sma10_up'] <= 12 and
        ((last_close - last['upper_band']) / last['upper_band']) >= 0.01 and
        last_close > last['span1'] and last_close > last['span2'] and
        prev_close > prev['sma20'] and
        last_15_bbw_min <= 3 and
        last_close > last['sma120'] and
        current_body_return >= previous_body_return and
        current_body_return >= 1 and
        recent_10_green < 9 and 
        (df['sma120_up'].tail(120).sum() <= 100)
    )
    if not uptrend:
        return False

    # === 추가 필터: sma5_down_10, bbw_narrowed_5days, etc
    sma5_down_10 = (calculate_consecutive(~df['sma5_up'].iloc[-10:]).max() >= 2)
    if not sma5_down_10:
        return False

    bbw_consecutive = calculate_consecutive(df['bbw_down'].iloc[-20:])
    if not (bbw_consecutive >= 5).any():
        return False

    sma20_down_15 = calculate_consecutive(df['sma20_down'].iloc[-15:])
    if sma20_down_15.max() >= 12:
        return False

    # 70봉 고점 돌파
    if not is_close_above_70_high(df):
        return False

    # === [추가 조건] 전/전전일 70봉 최고가
    try:
        prev2 = df.iloc[-3]
        prev_close2 = prev2['close']

        max70_prev  = df['close'].iloc[-72:-2].max()
        max70_prev2 = df['close'].iloc[-73:-3].max()

        is_prev70high   = (prev_close >= max70_prev)
        is_prev2_70high = (prev_close2 >= max70_prev2)

        body_current_rel = abs((last_close - last_open) / last_open)

        if is_prev70high:
            prev_body = abs((prev_close - prev_open) / prev_open)
            if prev_body > 0 and body_current_rel >= 3 * prev_body:
                return False

        if is_prev2_70high:
            prev2_open = prev2['open']
            prev2_body = abs((prev_close2 - prev2_open) / prev2_open)
            if prev2_body > 0 and body_current_rel >= 3 * prev2_body:
                return False

    except Exception as e:
        logging.info(f"[Check Prev70High Error] {sym}, {e}")
        return False

    # === [새 프랙탈 로직] "직전보다 더 낮은 프랙탈이 5번 이상이면 제외"
    try:
        # 최근 120봉의 프랙탈만 확인
        frac_values = df['fractal_low5_value'].iloc[-120:].dropna().values
        if len(frac_values) >= 2:
            lower_low_count = 0
            for i in range(1, len(frac_values)):
                if frac_values[i] < frac_values[i-1]:
                    lower_low_count += 1
            # 5번 이상이면 제외
            if lower_low_count >= 5:
                return False
    except Exception as e:
        logging.info(f"[Williams Fractal Error] {sym}, {e}")
        return False

    # === 일봉 Bollinger Band 조건
    try:
        daily_data = exchange.fetch_ohlcv(sym, timeframe='1d', limit=21)
        df_daily = pd.DataFrame(daily_data, columns=['timestamp','open','high','low','close','volume'])
        df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp'], unit='ms')
        df_daily.set_index('timestamp', inplace=True)
        df_daily[['open','high','low','close','volume']] = df_daily[['open','high','low','close','volume']].apply(pd.to_numeric)
        if len(df_daily) < 20:
            return False

        sma20_d = df_daily['close'].rolling(20).mean()
        std_d   = df_daily['close'].rolling(20).std()
        upper_d = sma20_d + 2 * std_d
        last_daily_close = df_daily['close'].iloc[-1]
        last_daily_upper = upper_d.iloc[-1]
        ratio = last_daily_close / last_daily_upper
        if 0.95 <= ratio <= 1.0:
            return False

    except Exception as e:
        logging.info(f"[Daily Data Error - Uptrend] {sym}, {e}")
        return False

    return True


def check_downtrend(df: pd.DataFrame, sym) -> bool:
    """다운트렌드: 기존 로직 그대로 (프랙탈 조건 미적용 가정)"""
    if len(df) < 120:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]
    last_close = last['close']
    last_open  = last['open']
    current_body_return = abs((last_close - last_open) / last_open) * 100

    prev_close = prev['close']
    prev_open  = prev['open']
    previous_body_return = abs((prev_close - prev_open) / prev_open) * 100

    last_20_bbw = df['bbw'].iloc[-20:]
    last_15_bbw_min = last_20_bbw.tail(15).min()
    recent_10_red = df['red_candle'].iloc[-10:].sum()

    if last['candle_body'] > 10 * last['avg_candle_body_20']:
        return False

    downtrend = (
        last_close <= last['lowest_120'] and
        ((last['highest_120'] - last_close) / last_close) <= 2 and
        last['volume'] > last['sma_volume20'] and
        last['consecutive_reds'] < 8 and
        last['sma10'] < last['sma20'] and
        prev['sma10'] < prev['sma20'] and
        last_close < last['lower_band'] and
        last['upper_band'] > prev['upper_band'] and
        last['lower_band'] < prev['lower_band'] and
        ((last['lower_band'] - last_close) / last_close) >= 0.01 and
        last_close < last['span1'] and last_close < last['span2'] and
        prev_close < prev['sma20'] and
        last_15_bbw_min <= 3 and
        last_close < last['sma120'] and
        current_body_return >= previous_body_return and
        last['consecutive_sma10_lt_sma20'] < 20 and
        current_body_return >= 1 and
        (df['sma120_down'].tail(120).sum() <= 100)
    )
    if not downtrend:
        return False

    # [추가 다운필터] sma5_up_10, etc.
    sma5_up_10 = (calculate_consecutive(df['sma5_up'].iloc[-10:]).max() >= 2)
    if not sma5_up_10:
        return False

    bbw_consecutive = calculate_consecutive(df['bbw_down'].iloc[-20:])
    if not (bbw_consecutive >= 5).any():
        return False

    sma20_up_15 = calculate_consecutive(df['sma20_up'].iloc[-15:])
    if sma20_up_15.max() >= 12:
        return False

    try:
        prev2 = df.iloc[-3]
        prev2_close = prev2['close']

        min70_prev  = df['close'].iloc[-72:-2].min()
        min70_prev2 = df['close'].iloc[-73:-3].min()

        is_prev70low  = (prev_close <= min70_prev)
        is_prev2_70low= (prev2_close <= min70_prev2)

        body_current_rel = abs((last_close - last_open) / last_open)

        if is_prev70low:
            prev_body = abs((prev_close - prev_open) / prev_open)
            if prev_body > 0 and body_current_rel >= 3 * prev_body:
                return False

        if is_prev2_70low:
            prev2_open = prev2['open']
            body_prev2 = abs((prev2_close - prev2_open) / prev2_open)
            if body_prev2 > 0 and body_current_rel >= 3 * body_prev2:
                return False

    except Exception as e:
        logging.info(f"[Check Prev70Low Error] {sym}, {e}")
        return False

    if not is_close_below_70_low(df):
        return False

    return True


##################################################
# 2) 포지션 & 매매 함수
##################################################
def get_open_positions():
    balance = exchange.fetch_balance(params={"type":"future"})
    positions = balance['info']['positions']
    open_positions = []
    for p in positions:
        amt = float(p['positionAmt'])
        if amt != 0:
            open_positions.append(p)
    return open_positions

def close_position_market(symbol, side, qty):
    order_side = 'sell' if side=='long' else 'buy'
    try:
        exchange.create_order(
            symbol=symbol,
            type='market',
            side=order_side,
            amount=qty
        )
        logging.info(f"[Close {side}] {symbol} qty={qty}")
    except Exception as e:
        logging.info(f"[Close Error] {symbol}, side={side}, {e}")

def set_leverage(symbol, lev=2):
    try:
        exchange.set_leverage(leverage=lev, symbol=symbol)
    except Exception as e:
        logging.info(f"[Leverage Error] {symbol}, {e}")

def open_position_with_tp(symbol, side, entry_price, capital_fraction=0.95, leverage=2, tp_percent=6.0):
    bal = exchange.fetch_balance(params={"type": "future"})
    usdt_info = bal.get('USDT', {})
    free_usdt = float(usdt_info.get('free', 0))

    invest = free_usdt * capital_fraction
    if invest < 5:
        logging.info(f"[Warning] Not enough USDT: {invest:.2f}, skip {symbol} {side}")
        return

    set_leverage(symbol, lev=leverage)

    qty = invest / entry_price
    qty = float(f"{qty:.4f}")

    order_side = 'buy' if side=='long' else 'sell'
    try:
        exchange.create_order(
            symbol=symbol,
            type='market',
            side=order_side,
            amount=qty
        )
        logging.info(f"[Open {side}] {symbol}, price={entry_price:.4f}, qty={qty}, invest={invest:.2f}")
    except Exception as e:
        logging.info(f"[Open Error] {symbol}, side={side}, {e}")
        return

    # 익절 주문
    if side=='long':
        tp_price = entry_price * (1 + tp_percent/100)
        tp_side = 'sell'
    else:
        tp_price = entry_price * (1 - tp_percent/100)
        tp_side = 'buy'

    try:
        exchange.create_order(
            symbol=symbol,
            type='TAKE_PROFIT_MARKET',
            side=tp_side,
            amount=qty,
            params={
                "stopPrice": float(f"{tp_price:.4f}"),
                "closePosition": True
            }
        )
        logging.info(f"[Set TP] {symbol}, side={side}, tp={tp_price:.4f} (+{tp_percent}%)")
    except Exception as e:
        logging.info(f"[TP Error] {symbol}, side={side}, {e}")


##################################################
# 3) "전거래일 대비 등락률" 계산
##################################################
def get_previous_day_return(symbol):
    df_daily = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=2)
    if len(df_daily) < 2:
        return None
    prev_close = df_daily[-2][4]
    last_close = df_daily[-1][4]
    if prev_close == 0:
        return None
    return (last_close - prev_close) / prev_close

##################################################
# 4) 심볼 스캔(멀티스레드)
##################################################
def analyze_symbol(sym):
    try:
        df = fetch_ohlcv_data(sym, timeframe='1h', limit=140)
        df = calculate_indicators(df)
        up = check_uptrend(df, sym)
        dn = check_downtrend(df, sym)
        return (sym if up else None, sym if dn else None)
    except Exception as e:
        logging.info(f"[analyze_symbol Error] {sym}, {e}")
        return (None, None)

##################################################
# 5) 메인 실행
##################################################
def main():
    logging.info("\n=== Start Auto-Trading ===")

    # (1) 현재 포지션 확인 -> 손절 조건(롱= 종가<20SMA, 숏= 종가>20SMA)
    open_pos = get_open_positions()
    have_long = False
    have_short = False

    for pos in open_pos:
        symbol = pos['symbol']
        amt = float(pos['positionAmt'])
        side = 'long' if amt > 0 else 'short'

        # 20SMA 확인(1h)
        try:
            df_1h = fetch_ohlcv_data(symbol, '1h', 30)
            df_1h = calculate_indicators(df_1h)
            latest_close = df_1h['close'].iloc[-1]
            latest_sma20 = df_1h['sma20'].iloc[-1]
        except Exception as e:
            logging.info(f"[Data Error] {symbol}, {e}")
            continue

        # 손절(간단 로직)
        if side == 'long' and latest_close < latest_sma20:
            close_position_market(symbol, side='long', qty=abs(amt))
        elif side == 'short' and latest_close > latest_sma20:
            close_position_market(symbol, side='short', qty=abs(amt))
        else:
            # 포지션 유지
            if side == 'long':
                have_long = True
            else:
                have_short = True

        if have_long or have_short:
            logging.info("현재 포지션 보유 중이므로 신규 진입을 진행하지 않습니다.")
            return

    # (2) 모든 선물 심볼 가져오기
    try:
        info = client.futures_exchange_info()
        all_symbols = [item['symbol'] for item in info['symbols']
                       if item['status'] == 'TRADING'
                       and item['symbol'].endswith('USDT')]
    except Exception as e:
        logging.info(f"[Load Markets Error] {e}")
        return

    logging.info(f"[All Futures Symbols] {len(all_symbols)}개")

    # (3) 멀티쓰레드 스캔
    up_candidates = []
    down_candidates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {executor.submit(analyze_symbol, sym): sym for sym in all_symbols}
        for fut in concurrent.futures.as_completed(future_map):
            sym = future_map[fut]
            try:
                up, down = fut.result()
                if up:
                    up_candidates.append(up)
                if down:
                    down_candidates.append(down)
            except:
                pass

    # (4) 전일 대비 변동률 계산
    best_up_sym = None
    best_up_return = -99999
    for sym in up_candidates:
        try:
            r = get_previous_day_return(sym)
            if r is not None and r > best_up_return:
                best_up_return = r
                best_up_sym = sym
        except:
            pass

    best_down_sym = None
    best_down_return = 99999
    for sym in down_candidates:
        try:
            r = get_previous_day_return(sym)
            if r is not None and r < best_down_return:
                best_down_return = r
                best_down_sym = sym
        except:
            pass

    logging.info(f"[Uptrend Candidates] {up_candidates}")
    logging.info(f"[Downtrend Candidates] {down_candidates}")
    logging.info(f"  => Best Up Symbol: {best_up_sym} (전일 대비 상승률={best_up_return*100:.2f}%)")
    logging.info(f"  => Best Down Symbol: {best_down_sym} (전일 대비 하락률={best_down_return*100:.2f}%)")

    # (5) 신규 진입
    up_sym = best_up_sym
    dn_sym = best_down_sym

    if (not have_long) and (not have_short) and (up_sym is not None) and (dn_sym is not None):
        cap_frac = 0.45
        # 롱
        try:
            df_up = fetch_ohlcv_data(up_sym, '1h', 2)
            price_up = df_up['close'].iloc[-1]
            open_position_with_tp(up_sym, side='long', entry_price=price_up,
                                  capital_fraction=cap_frac, leverage=2, tp_percent=5.0)
            have_long = True
        except:
            pass
        # 숏
        try:
            df_dn = fetch_ohlcv_data(dn_sym, '1h', 2)
            price_dn = df_dn['close'].iloc[-1]
            open_position_with_tp(dn_sym, side='short', entry_price=price_dn,
                                  capital_fraction=cap_frac, leverage=2, tp_percent=5.0)
            have_short = True
        except:
            pass
    else:
        # Uptrend만
        if (up_sym is not None) and (not have_long):
            cap_frac = 0.90 if not have_short else 0.45
            try:
                df_up = fetch_ohlcv_data(up_sym, '1h', 2)
                price_up = df_up['close'].iloc[-1]
                open_position_with_tp(up_sym, side='long', entry_price=price_up,
                                      capital_fraction=cap_frac, leverage=2, tp_percent=5.0)
                have_long = True
            except:
                pass

        # Downtrend만
        if (dn_sym is not None) and (not have_short):
            cap_frac = 0.90 if not have_long else 0.45
            try:
                df_dn = fetch_ohlcv_data(dn_sym, '1h', 2)
                price_dn = df_dn['close'].iloc[-1]
                open_position_with_tp(dn_sym, side='short', entry_price=price_dn,
                                      capital_fraction=cap_frac, leverage=2, tp_percent=5.0)
                have_short = True
            except:
                pass

    logging.info("=== End Auto-Trading ===")

if __name__ == "__main__":
    main()

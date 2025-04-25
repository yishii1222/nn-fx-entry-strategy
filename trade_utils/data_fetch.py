import time
import pandas as pd
from datetime import timedelta
from dateutil import parser
import requests
from .config import ACCESS_TOKEN, INSTRUMENT


def fetch_1min_data(start: pd.Timestamp, end: pd.Timestamp, access_token: str, instrument: str) -> pd.DataFrame:
    url     = f'https://api-fxtrade.oanda.com/v3/instruments/{instrument}/candles'
    headers = {'Authorization': f'Bearer {access_token}'}
    current = start
    all_data = []
    while current < end:
        params = {
            'from':        current.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'granularity': 'M1',
            'count':       500,
            'price':       'M'
        }
        try:
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json().get('candles', [])
        except Exception as e:
            print(f"Fetch error: {e}")
            break

        valid = [c for c in data if c.get('complete')]
        if not valid:
            current += timedelta(minutes=1)
            continue

        for c in valid:
            all_data.append({
                'time':   c['time'],
                'open':   float(c['mid']['o']),
                'high':   float(c['mid']['h']),
                'low':    float(c['mid']['l']),
                'close':  float(c['mid']['c']),
                'volume': int(c['volume'])
            })

        current = parser.isoparse(valid[-1]['time']) + timedelta(minutes=1)
        time.sleep(0.2)

    df = pd.DataFrame(all_data)
    if df.empty:
        return df
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('time', inplace=True)
    return df
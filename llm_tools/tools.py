from langchain_core.tools import tool
import time
from dotenv import load_dotenv
import os
import requests

load_dotenv(verbose=True)

GD_WEATHER_API = os.getenv("GD_WEATHER_API")
GD_WEATHER_KEY = os.getenv("GD_WEATHER_KEY")
YAYA_SEARCH_URL = os.getenv('YAYA_SEARCH_URL')

@tool
def get_datetime():
    """
    获取当前系统日期时间
    :return: 日期时间字符串，例：2026-09-01 15:00:00
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

@tool
def get_weather(city:str, extensions:str):
    """
    获取指定城市天气
    :param city: 城市
    :param extensions: base:返回现在实时天气 all:返回预报天气
    :return: 返回天气信息
    """
    response = requests.get(
        url=GD_WEATHER_API,
        params={
            "key": GD_WEATHER_KEY,
            "city": city,
            "extensions": extensions,
        }
    )
    weather_info = response.json()['lives']
    return weather_info


def web_search(keyword:str):
    """
    使用网络搜索
    :param keyword: 搜索关键字
    :return: 搜索此关键字得到的结果
    """
    response = requests.get(
        url=YAYA_SEARCH_URL,
        params={
            "ia": 'api',
            "origin": "funnel_home_searchresults",
            "q": keyword,
        }
    )
    return response.content

if __name__ == '__main__':
    print(web_search("rwkv llm"))
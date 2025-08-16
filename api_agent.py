from fastapi import FastAPI, HTTPException, Query
import yfinance as yf
import requests
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("FINNHUB_API_KEY")

app = FastAPI()

RETRIEVER_URL = "http://localhost:8004/add_documents"

def push_to_retriever(docs: list[str]):
    try:
        response = requests.post(RETRIEVER_URL, json={"docs": docs})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/marketdata/{ticker}")
async def get_market_data(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if 'regularMarketPrice' not in info or info['regularMarketPrice'] is None:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found or no market data available.")
        
        data = {
            "ticker": ticker.upper(),
            "current_price": info.get("regularMarketPrice"),
            "previous_close": info.get("previousClose"),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap"),
        }
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/earnings/{ticker}")
async def get_earnings(ticker: str):
    try:
        url = f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker.upper()}&token={API_KEY}"
        response = requests.get(url)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch earnings data.")
        
        earnings = response.json()
        if not earnings:
            return {"data": {"ticker": ticker.upper(), "message": "No earnings data found."}}

        latest = earnings[0]
        return {
            "data": {
                "ticker": latest.get("symbol", ticker.upper()),
                "date": latest.get("period", "N/A"),
                "epsActual": latest.get("actual", "N/A"),
                "epsEstimate": latest.get("estimate", "N/A"),
                "surprisePercent": latest.get("surprisePercent", "N/A")
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historical/{ticker}")
async def get_historical_data(
    ticker: str, 
    start: str = Query(None),
    end: str = Query(None)
):
    try:
        stock = yf.Ticker(ticker)
        start_date = datetime.strptime(start, "%Y-%m-%d") if start else None
        end_date = datetime.strptime(end, "%Y-%m-%d") if end else None
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return {"ticker": ticker.upper(), "data": [], "message": "No historical data found."}

        data = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "volume": int(row['Volume']),
            }
            for date, row in hist.iterrows()
        ]
        return {"ticker": ticker.upper(), "data": data}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/push_to_retriever/{ticker}")
async def push_api_data(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if 'regularMarketPrice' not in info:
            raise HTTPException(status_code=404, detail="Ticker not found or no data.")

        summary = (
            f"{ticker.upper()} current price: {info.get('regularMarketPrice')} USD. "
            f"Prev close: {info.get('previousClose')}, high: {info.get('dayHigh')}, "
            f"low: {info.get('dayLow')}, volume: {info.get('volume')}, market cap: {info.get('marketCap')}."
        )
        result = push_to_retriever([summary])
        return {"summary": summary, "retriever_response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

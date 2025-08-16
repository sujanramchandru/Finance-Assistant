import pandas as pd
import yfinance as yf
import math
import requests
from fastapi import FastAPI

app = FastAPI()

RETRIEVER_URL = "http://localhost:8004/add_documents"

def push_to_retriever(docs: list[str]):
    try:
        response = requests.post(RETRIEVER_URL, json={"docs": docs})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def fetch_earnings_data(tickers):
    results = []
    for ticker in tickers:
        company_data = {"ticker": ticker}
        try:
            stock = yf.Ticker(ticker)
            income_stmt = stock.quarterly_income_stmt
            if income_stmt is None or income_stmt.empty:
                company_data['quarterly_net_income'] = None
            else:
                net_income_series = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None
                company_data['quarterly_net_income'] = net_income_series.to_dict() if net_income_series is not None else None
            company_data['earnings_dates'] = []
        except Exception as e:
            company_data['error'] = str(e)
        results.append(company_data)
    return results

def clean_data(results):
    for entry in results:
        net_income = entry.get('quarterly_net_income')
        if isinstance(net_income, dict):
            cleaned_net_income = {}
            for k, v in net_income.items():
                key_str = k.strftime('%Y-%m-%d') if hasattr(k, 'strftime') else str(k)
                value = None if (isinstance(v, float) and math.isnan(v)) else v
                cleaned_net_income[key_str] = value
            entry['quarterly_net_income'] = cleaned_net_income
    return results

def generate_summary(entry):
    ticker = entry.get("ticker")
    ni = entry.get("quarterly_net_income")
    if not ni:
        return f"{ticker} has no recent net income data."
    return f"{ticker} net income: " + ", ".join([f"{d}: {v}" for d, v in ni.items()])

@app.post("/push_scraped_data/{ticker}")
def push_scraped_data(ticker: str):
    raw = fetch_earnings_data([ticker])
    cleaned = clean_data(raw)
    summaries = [generate_summary(entry) for entry in cleaned]
    result = push_to_retriever(summaries)
    return {"summaries": summaries, "retriever_response": result}

import re
import httpx
import spacy
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Agent URLs
API_AGENT_URL = "http://127.0.0.1:8002"
SCRAPING_AGENT_URL = "http://127.0.0.1:8003"
RETRIEVER_AGENT_URL = "http://127.0.0.1:8004"
LLM_AGENT_URL = "http://127.0.0.1:8005"

nlp = spacy.load("en_core_web_sm")

class TranscriptionRequest(BaseModel):
    transcription: str

# Ticker pattern (1–5 uppercase letters)
TICKER_REGEX = r'\b[A-Z]{1,5}\b'

# Intent keywords map
INTENT_KEYWORDS = {
    "earnings": ["earnings", "earnings report", "report"],
    "historical": ["history", "historical", "past", "previous"],
    "price": ["price", "stock", "quote", "current price"]
}


def extract_tickers(text):
    return re.findall(TICKER_REGEX, text)


def find_intent_for_phrase(phrase: str):
    phrase = phrase.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in phrase for kw in keywords):
            return intent
    return "price"  # Default intent


def parse_query(user_text: str):
    doc = nlp(user_text)
    chunks = []
    current_chunk = []

    for token in doc:
        if token.text.lower() == "and":
            if current_chunk:
                chunks.append(" ".join([t.text for t in current_chunk]))
                current_chunk = []
        else:
            current_chunk.append(token)

    if current_chunk:
        chunks.append(" ".join([t.text for t in current_chunk]))

    ticker_intent_map = {}
    for chunk in chunks:
        tickers = extract_tickers(chunk)
        if not tickers:
            continue
        intent = find_intent_for_phrase(chunk)
        for t in tickers:
            ticker_intent_map[t] = intent

    return ticker_intent_map


async def fetch_data_for_ticker(client, ticker: str, intent: str):
    if intent == "price":
        api_resp = await client.get(f"{API_AGENT_URL}/marketdata/{ticker}")
        return api_resp.json()

    elif intent == "earnings":
        api_resp = await client.get(f"{API_AGENT_URL}/earnings/{ticker}")
        scraping_resp = await client.post(f"{SCRAPING_AGENT_URL}/push_scraped_data/{ticker}")
        return {
            "api_earnings": api_resp.json(),
            "scraping": scraping_resp.json()
        }

    elif intent == "historical":
        api_resp = await client.get(f"{API_AGENT_URL}/historical/{ticker}?start=2025-04-01&end=2025-05-01")
        return api_resp.json()

    return {}


def build_context(agent_data, ticker):
    """Builds a plain string context from agent data per ticker."""
    context = f"{ticker} info:\n"
    context += str(agent_data.get(ticker))
    return context


@app.post("/receive_transcription")
async def receive_transcription(data: TranscriptionRequest):
    user_text = data.transcription
    print("Received transcription:", user_text)

    ticker_intent_map = parse_query(user_text)
    print("Parsed ticker-intent map:", ticker_intent_map)

    responses = {}
    documents_to_add = []

    async with httpx.AsyncClient() as client:
        # Step 1: Fetch data and push to retriever
        for ticker, intent in ticker_intent_map.items():
            data = await fetch_data_for_ticker(client, ticker, intent)
            responses[ticker] = data
            documents_to_add.append(str(data))

        if documents_to_add:
            add_docs_resp = await client.post(
                f"{RETRIEVER_AGENT_URL}/add_documents",
                json={"docs": documents_to_add}
            )
            print("Retriever add_documents response:", add_docs_resp.text)

        # Step 2: Query retriever
        retriever_payload = {
            "query": user_text,
            "top_k": 1
        }
        retriever_query_resp = await client.post(
            f"{RETRIEVER_AGENT_URL}/query",
            json=retriever_payload
        )
        retriever_result = retriever_query_resp.json() if retriever_query_resp.status_code == 200 else {"error": retriever_query_resp.text}

        # Step 3: Build full context for LLM
        full_context = "\n\n".join([build_context(responses, ticker) for ticker in ticker_intent_map])

        llm_payload = {
            "user_query": user_text,
            "retrieved_docs": [full_context]
        }

        # Step 4: Call LLM agent
        llm_response = await client.post(f"{LLM_AGENT_URL}/generate/", json=llm_payload, timeout=300.0)
        llm_result = llm_response.json() if llm_response.status_code == 200 else {"error": llm_response.text}
        # Send the LLM response to TTS agent
        tts_text = llm_result.get("response", "Sorry, I don't have an answer.")
        tts_payload = {"text": tts_text}

        tts_response = await client.post("http://127.0.0.1:8006/speak", json=tts_payload)
        tts_result = tts_response.json() if tts_response.status_code == 200 else {"error": tts_response.text}


    return {
        "user_query": user_text,
        "ticker_intent_map": ticker_intent_map,
        "agent_data": responses,
        "retriever_result": retriever_result,
        "llm_response": llm_result
    }

import os
import requests
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from streamlit_autorefresh import st_autorefresh


st.set_page_config(page_title="AeroStream Sentiment", layout="centered")


def _api_base_url() -> str:
	url = (os.getenv("FASTAPI_URL") or "http://localhost:8000").strip()
	return url.rstrip("/")


def _database_url() -> str:
	url = os.getenv("DATABASE_URL", "").strip()
	if not url:
		raise RuntimeError("DATABASE_URL env var is missing (needed for KPI dashboard)")
	return url


@st.cache_resource(show_spinner=False)
def _db_engine():
	return create_engine(_database_url(), pool_pre_ping=True)


def _load_latest_kpis() -> dict | None:
	engine = _db_engine()
	with engine.connect() as conn:
		# Table is created by Airflow; if it doesn't exist yet, we just show a helpful message.
		try:
			row = conn.execute(
				text(
					"""
					SELECT computed_at, total_tweets, total_airlines, negative_pct
					FROM kpi_snapshots
					ORDER BY computed_at DESC
					LIMIT 1;
					"""
				)
			).mappings().first()
		except Exception:
			return None

		return dict(row) if row else None


def _load_recent_tweets(limit: int = 50) -> pd.DataFrame:
	engine = _db_engine()
	query = text(
		"""
		SELECT id, airline, prediction, airline_sentiment_confidence, negativereason, tweet_created, created_at, text
		FROM tweets
		ORDER BY created_at DESC NULLS LAST, id DESC
		LIMIT :limit;
		"""
	)
	try:
		with engine.connect() as conn:
			return pd.read_sql(query, conn, params={"limit": int(limit)})
	except Exception:
		return pd.DataFrame()


def _predict(api_url: str, text: str) -> dict:
	resp = requests.post(
		f"{api_url}/predict",
		json={"text": text},
		timeout=30,
	)
	resp.raise_for_status()
	return resp.json()


def _render_kpi_dashboard():
	st.subheader("Live KPI Dashboard")
	try:
		kpis = _load_latest_kpis()
		if not kpis:
			st.info("No KPI snapshot yet. Start Airflow and wait ~1 minute for the first run.")
		else:
			c1, c2, c3 = st.columns(3)
			c1.metric("Nombre total de tweets", int(kpis["total_tweets"]))
			c2.metric("Nombre de compagnies aériennes", int(kpis["total_airlines"]))
			c3.metric("Pourcentage de tweets négatifs", f"{float(kpis['negative_pct']):.2f}%")
			st.caption(f"Last updated: {kpis['computed_at']}")

		recent = _load_recent_tweets(limit=50)
		if recent.empty:
			st.warning("No tweets in database yet.")
		else:
			st.dataframe(recent, use_container_width=True, hide_index=True)
	except Exception as e:
		st.error(f"Dashboard error: {e}")


def _render_prediction():
	api_url = _api_base_url()
	st.link_button("API", "http://localhost:8000/docs")

	text = st.text_area(
		"Customer feedback",
		placeholder="Type a customer review...",
		height=140,
	)

	predict_clicked = st.button("Predict", type="primary", disabled=not text.strip())

	if predict_clicked:
		try:
			with st.spinner("Predicting..."):
				result = _predict(api_url, text)
			st.subheader("Result")
			st.write(result)
		except requests.HTTPError as e:
			body = getattr(e.response, "text", "")
			st.error(f"HTTP error: {e}\n\n{body}")
		except requests.RequestException as e:
			st.error(f"Request failed: {e}")


st.title("AeroStream – Sentiment Prediction")

st.sidebar.header("Navigation")
active_view = st.sidebar.radio(
	"Go to",
	("KPI Statistics", "Prediction"),
	index=0,
)

# Refresh the KPI dashboard once per minute (matches the Airflow schedule).
if active_view == "KPI Statistics":
	st_autorefresh(interval=60 * 1000, key="aerostream_autorefresh")
	_render_kpi_dashboard()
else:
	_render_prediction()
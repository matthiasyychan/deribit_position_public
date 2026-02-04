# Deribit Positions Dashboard

A comprehensive real-time dashboard for monitoring cryptocurrency trading positions on Deribit exchange. Built with Streamlit for easy deployment and interactive data visualization.

## Features

- 📊 **Real-time Position Tracking** - Live data from Deribit API
- 🔀 **Multi-Account Support** - Monitor multiple sub-accounts simultaneously
- 📈 **Advanced Metrics** - Track P&L, Greeks (delta, gamma, vega, theta), margins
- 💰 **Multi-Currency** - Support for BTC, ETH, and USDC-settled instruments
- 📥 **Export Options** - Download data in CSV, Excel, or JSON formats
- 🔍 **Flexible Filtering** - Filter by currency, instrument kind (options/futures)
- 🔒 **Secure Configuration** - Supports Streamlit secrets, environment variables, or local config

## Quick Start

Deploy to Streamlit Cloud or run locally with:
pip install -r requirements.txt
streamlit run streamlit_app.py

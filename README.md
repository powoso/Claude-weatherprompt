# Weather Edge

Find +EV opportunities on weather prediction markets (Polymarket, Kalshi) by comparing calibrated model probabilities against market prices.

## How It Works

1. **Ingest** 30+ years of historical weather data (Open-Meteo, HURDAT2 hurricanes, ENSO/ONI)
2. **Fit** statistical models — distributions, base rates, ENSO adjustments, analog year matching
3. **Simulate** — Monte Carlo for temperature, Poisson regression for hurricanes, logistic/bootstrap for snow
4. **Compare** model probabilities vs live market prices to find mispriced contracts
5. **Size** bets using half-Kelly criterion with configurable max position limits

## Install on macOS

### Prerequisites

- Python 3.10 or newer
- pip (included with Python)

If you don't have Python 3.10+, install it with [Homebrew](https://brew.sh):

```bash
brew install python@3.12
```

### Setup

```bash
# Clone the repo
git clone https://github.com/powoso/Claude-weatherprompt.git
cd Claude-weatherprompt

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package and all dependencies
pip install -e ".[dev]"

# Copy the environment file and add your API tokens
cp .env.example .env
```

Edit `.env` and add your NOAA token (free, takes 10 seconds):

```bash
# Get a free token at https://www.ncdc.noaa.gov/cdo-web/token
NOAA_CDO_TOKEN=your_token_here
```

Kalshi and alert webhook credentials are optional.

### Verify the install

```bash
# Run the test suite
pytest

# Check the CLI is available
weather-edge --help
```

## Usage

### 1. Ingest historical data

Downloads 30 years of daily weather for 10 major US cities, HURDAT2 hurricane records, and ENSO index data. Takes a few minutes on the first run.

```bash
weather-edge ingest
```

### 2. Scan markets for opportunities

Fetches live weather contracts from Polymarket and Kalshi, evaluates each against the model, and flags +EV opportunities.

```bash
weather-edge scan
```

Example output:

```
Found 2 opportunities:
  [polymarket] Will NYC exceed 100F in July 2026?
    Model: 12.3% | Market: 8.0% | Edge: +4.3% | Kelly bet: $21.50
  [kalshi] Snow in Austin before March 2026?
    Model: 45.2% | Market: 38.0% | Edge: +7.2% | Kelly bet: $36.00
```

### 3. Query base rates

Look up historical base rates for any city/metric/threshold combination:

```bash
weather-edge base-rates "New York City" -m temperature_2m_max -t 100 --month 7
weather-edge base-rates "Austin" -m snowfall_sum -t 0.1 --month 1
weather-edge base-rates "Miami" -m precipitation_sum -t 2.0 --month 9
```

### 4. Launch the dashboard

Interactive Streamlit dashboard with five tabs: active opportunities, historical base rate explorer, ENSO status, model calibration, and bankroll management.

```bash
weather-edge dashboard
```

Opens at `http://localhost:8501`.

### 5. Run continuously

Scans markets every hour and runs a full model update every 6 hours. Sends alerts via Slack/Discord/email when new opportunities appear.

```bash
weather-edge cron
```

## Configuration

All settings live in `config.yaml`:

| Section | What it controls |
|---|---|
| `cities` | Which cities to track (lat/lon, NWS gridpoints, UHI offsets) |
| `probability` | Monte Carlo sim count, bootstrap samples, distribution choices |
| `edge` | Minimum edge threshold, Kelly fraction, max position size, bankroll |
| `alerts` | Email/Slack/Discord webhook settings and alert thresholds |
| `dashboard` | Host and port for Streamlit |

Add or remove cities by editing the `cities` list. Each city needs:

```yaml
- name: "Seattle"
  state: "WA"
  lat: 47.6062
  lon: -122.3321
  nws_office: "SEW"
  nws_gridpoint: "124,67"
  noaa_station: "USW00024233"
  uhi_adjustment_f: 1.5
```

## Where the Edge Comes From

Markets tend to be least efficient on:

- **Seasonal contracts** — "Will X happen this season?" gets mispriced because traders anchor to recent memory, not base rates
- **ENSO-sensitive events** — El Nino/La Nina dramatically shifts hurricane, temperature, and snowfall probabilities, but markets often ignore this
- **Tail events in unusual cities** — "Snow in Austin" or "100F in NYC" have real historical base rates that differ from gut estimates
- **Long-dated contracts** — more time = more uncertainty = more room for the model to disagree with the crowd

## Project Structure

```
weather_edge/
  data_collection/    Historical data, forecasts, ENSO, market scraping
  features/           Base rates, distribution fitting, analog years, engineering
  probability/        Monte Carlo temp, Poisson hurricane, logistic snow, engine
  edge/               Edge detection, Kelly sizing, calibration tracking
  alerts/             Email/Slack/Discord notifications
  dashboard/          Streamlit app
  pipeline.py         Orchestrates ingest -> scan -> update flows
  cli.py              Click CLI entry point
```

## Running Tests

```bash
pytest                  # all tests
pytest -v               # verbose
pytest tests/test_probability.py  # single module
```

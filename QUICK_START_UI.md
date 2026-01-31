# Quick Start Guide - UI

## Prerequisites

- Python 3.8+
- Node.js 18+ and npm
- macOS (for start scripts)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install UI Dependencies

```bash
cd ui
npm install
cd ..
```

## Starting the Application

### Option 1: Use Start Script (Recommended)

The easiest way to start both Flask API and React UI:

```bash
# Make scripts executable (first time only)
chmod +x start.sh start-dev.sh

# Standard mode (background processes, logs to files)
./start.sh

# Development mode (separate terminal windows)
./start-dev.sh
```

### Option 2: Manual Start

#### Terminal 1 - Flask API:
```bash
python run_server.py
```

#### Terminal 2 - React UI:
```bash
cd ui
npm run dev
```

## Access Points

Once started, access the application at:

- **React UI**: http://localhost:3000
- **Flask API**: http://localhost:5000
- **API Docs (Swagger)**: http://localhost:5000/api/docs

## Features

### Dashboard
- System health status
- Available models count
- Active cron jobs
- Quick action cards

### Predictions
- Single prediction for a symbol
- Multi-step predictions (t+1 to t+4)
- Batch predictions for multiple symbols

### Models
- View all model versions
- Manage PROD versions
- Models grouped by timeframe

### Forward Testing
- Monitor cron job status
- View registered jobs
- Manual trigger for pending tests
- View test results

### History
- Query historical regime data
- Interactive charts
- Filter by symbol and timeframe
- Query by hours or date range

## Troubleshooting

### Port Already in Use

If ports 5000 or 3000 are already in use:

```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Dependencies Not Found

If you see import errors:

```bash
# Reinstall Python dependencies
pip install -r requirements.txt

# Reinstall UI dependencies
cd ui
npm install
cd ..
```

### Script Permission Denied

```bash
chmod +x start.sh start-dev.sh
```

## Stopping Services

Press `Ctrl+C` in the terminal where you ran the start script, or:

```bash
# Kill Flask API
pkill -f "python3 run_server.py"

# Kill React UI
pkill -f "npm run dev"
pkill -f "vite"
```

## Development

### UI Development

The UI uses Vite for fast hot-reload development. Changes to React components will automatically reload.

### API Development

The Flask API runs with auto-reload enabled. Changes to Python files will automatically restart the server.

## Environment Variables

### UI

Set `VITE_API_URL` to change the API endpoint:

```bash
VITE_API_URL=http://your-api-url:port/api npm run dev
```

### Flask API

Configure via `config.py` or environment variables.

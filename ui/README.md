# Regime Trainer UI

A modern React-based user interface for the Regime Trainer API.

## Features

- 🎨 Modern, responsive UI with Tailwind CSS
- 📊 Real-time predictions and visualizations
- 🔄 Model version management
- 🧪 Forward testing monitoring
- 📈 Historical data visualization
- ⚡ Fast development with Vite

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm
- Flask API server running on `http://localhost:5858`

### Installation

```bash
cd ui
npm install
```

### Development

Start the development server:

```bash
npm run dev
```

The UI will be available at `http://localhost:3000`

### Build

Build for production:

```bash
npm run build
```

Preview production build:

```bash
npm run preview
```

## Configuration

The UI connects to the Flask API by default at `http://localhost:5858`. To change this, set the `VITE_API_URL` environment variable:

```bash
VITE_API_URL=http://your-api-url:port/api npm run dev
```

## Project Structure

```
ui/
├── src/
│   ├── api/          # API client and services
│   ├── components/   # Reusable components
│   ├── pages/        # Page components
│   ├── App.tsx       # Main app component
│   └── main.tsx      # Entry point
├── public/           # Static assets
└── package.json      # Dependencies
```

## Pages

- **Dashboard**: Overview of system status and quick actions
- **Predictions**: Single, multi-step, and batch predictions
- **Models**: Model version management and PROD settings
- **Forward Testing**: Monitor cron jobs and trigger tests
- **History**: View historical regime data with charts

## Technologies

- React 18
- TypeScript
- Vite
- Tailwind CSS
- React Router
- Axios
- Recharts
- Lucide React (icons)

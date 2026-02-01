import { useState } from 'react'
import { Play, BarChart3 } from 'lucide-react'
import {
  predictNextRegime,
  predictRegimes,
  batchPredict,
  type PredictionResult,
  type MultiStepPrediction,
} from '../api/services'

const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
const TIMEFRAMES = ['5m', '15m', '1h']

export default function Predictions() {
  const [symbol, setSymbol] = useState('BTCUSDT')
  const [timeframe, setTimeframe] = useState('15m')
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [multiStep, setMultiStep] = useState<MultiStepPrediction | null>(null)
  const [batchResults, setBatchResults] = useState<Record<string, PredictionResult> | null>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'single' | 'multi' | 'batch'>('single')

  const handleSinglePredict = async () => {
    setLoading(true)
    try {
      const result = await predictNextRegime(symbol, timeframe)
      setPrediction(result)
    } catch (error) {
      console.error('Prediction failed:', error)
      alert('Failed to get prediction')
    } finally {
      setLoading(false)
    }
  }

  const handleMultiStepPredict = async () => {
    setLoading(true)
    try {
      const result = await predictRegimes(symbol, timeframe, true)
      setMultiStep(result)
    } catch (error) {
      console.error('Multi-step prediction failed:', error)
      alert('Failed to get multi-step prediction')
    } finally {
      setLoading(false)
    }
  }

  const handleBatchPredict = async () => {
    setLoading(true)
    try {
      const result = await batchPredict(SYMBOLS, timeframe)
      setBatchResults(result)
    } catch (error) {
      console.error('Batch prediction failed:', error)
      alert('Failed to get batch predictions')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h1 className="text-3xl font-bold text-white mb-8">Predictions</h1>

      {/* Controls */}
      <div className="card mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Symbol
            </label>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="input-field w-full"
            >
              {SYMBOLS.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Timeframe
            </label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="input-field w-full"
            >
              {TIMEFRAMES.map((tf) => (
                <option key={tf} value={tf}>
                  {tf}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setActiveTab('single')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              activeTab === 'single'
                ? 'bg-primary-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Single Prediction
          </button>
          <button
            onClick={() => setActiveTab('multi')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              activeTab === 'multi'
                ? 'bg-primary-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Multi-Step
          </button>
          <button
            onClick={() => setActiveTab('batch')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              activeTab === 'batch'
                ? 'bg-primary-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Batch
          </button>
        </div>

        <button
          onClick={
            activeTab === 'single'
              ? handleSinglePredict
              : activeTab === 'multi'
              ? handleMultiStepPredict
              : handleBatchPredict
          }
          disabled={loading}
          className="btn-primary flex items-center gap-2"
        >
          <Play size={20} />
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </div>

      {/* Results */}
      {activeTab === 'single' && prediction && (
        <div className="card">
          <h2 className="text-xl font-semibold text-white mb-4">Prediction Result</h2>
          <div className="space-y-4">
            <div>
              <p className="text-slate-400 mb-2">Most Likely Regime</p>
              <div className="bg-slate-700 p-4 rounded-lg">
                <p className="text-2xl font-bold text-white">{prediction.most_likely_regime.name}</p>
                <p className="text-slate-400">
                  Probability: {(prediction.most_likely_regime.probability * 100).toFixed(2)}%
                </p>
              </div>
            </div>
            <div>
              <p className="text-slate-400 mb-2">All Regimes</p>
              <div className="space-y-2">
                {(prediction.all_regimes || []).map((regime, idx) => (
                  <div key={idx} className="bg-slate-700 p-3 rounded-lg flex justify-between">
                    <span className="text-white">{regime.name}</span>
                    <span className="text-slate-400">
                      {(regime.probability * 100).toFixed(2)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'multi' && multiStep && (
        <div className="card">
          <h2 className="text-xl font-semibold text-white mb-4">Multi-Step Predictions</h2>
          <div className="space-y-4">
            {Array.isArray(multiStep.predictions) ? (
              multiStep.predictions.map((pred, idx) => (
                <div key={idx} className="bg-slate-700 p-4 rounded-lg">
                  <h3 className="text-lg font-medium text-white mb-2">Step {pred.step}</h3>
                  <div className="space-y-2">
                    {Array.isArray(pred.regimes) ? (
                      pred.regimes.map((regime, rIdx) => (
                        <div key={rIdx} className="flex justify-between">
                          <span className="text-slate-300">{regime.name}</span>
                          <span className="text-slate-400">
                            {(regime.probability * 100).toFixed(2)}%
                          </span>
                        </div>
                      ))
                    ) : (
                      <p className="text-slate-400">No regime data available</p>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <p className="text-slate-400">No predictions available</p>
            )}
          </div>
        </div>
      )}

      {activeTab === 'batch' && batchResults && (
        <div className="card">
          <h2 className="text-xl font-semibold text-white mb-4">Batch Predictions</h2>
          <div className="space-y-4">
            {Object.entries(batchResults).map(([sym, pred]) => (
              <div key={sym} className="bg-slate-700 p-4 rounded-lg">
                <h3 className="text-lg font-medium text-white mb-2">{sym}</h3>
                <p className="text-primary-400 font-semibold">
                  {pred.most_likely_regime.name} ({(pred.most_likely_regime.probability * 100).toFixed(2)}%)
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

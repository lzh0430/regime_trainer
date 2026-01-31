import { useState } from 'react'
import { Search, Calendar } from 'lucide-react'
import { getHistory, type HistoryData } from '../api/services'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
const TIMEFRAMES = ['5m', '15m', '1h']

export default function History() {
  const [symbol, setSymbol] = useState('BTCUSDT')
  const [timeframe, setTimeframe] = useState('15m')
  const [history, setHistory] = useState<HistoryData | null>(null)
  const [loading, setLoading] = useState(false)
  const [queryType, setQueryType] = useState<'hours' | 'range'>('hours')
  const [lookbackHours, setLookbackHours] = useState(24)
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')

  const handleSearch = async () => {
    setLoading(true)
    try {
      const options: any = {}
      if (queryType === 'hours') {
        options.lookback_hours = lookbackHours
      } else {
        if (startDate) options.start_date = startDate
        if (endDate) options.end_date = endDate
      }
      const result = await getHistory(symbol, timeframe, options)
      setHistory(result)
    } catch (error) {
      console.error('Failed to fetch history:', error)
      alert('Failed to fetch history')
    } finally {
      setLoading(false)
    }
  }

  const chartData = history?.history.map((item) => ({
    time: new Date(item.timestamp).toLocaleString(),
    regime: item.regime,
  })) || []

  return (
    <div>
      <h1 className="text-3xl font-bold text-white mb-8">History</h1>

      {/* Search Controls */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold text-white mb-4">Query Parameters</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Symbol</label>
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
            <label className="block text-sm font-medium text-slate-300 mb-2">Timeframe</label>
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

        <div className="mb-4">
          <div className="flex gap-4 mb-4">
            <button
              onClick={() => setQueryType('hours')}
              className={`px-4 py-2 rounded-lg ${
                queryType === 'hours'
                  ? 'bg-primary-600 text-white'
                  : 'bg-slate-700 text-slate-300'
              }`}
            >
              By Hours
            </button>
            <button
              onClick={() => setQueryType('range')}
              className={`px-4 py-2 rounded-lg ${
                queryType === 'range'
                  ? 'bg-primary-600 text-white'
                  : 'bg-slate-700 text-slate-300'
              }`}
            >
              By Date Range
            </button>
          </div>

          {queryType === 'hours' ? (
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Lookback Hours
              </label>
              <input
                type="number"
                value={lookbackHours}
                onChange={(e) => setLookbackHours(parseInt(e.target.value))}
                className="input-field w-full"
                min="1"
              />
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Start Date
                </label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="input-field w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  End Date
                </label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="input-field w-full"
                />
              </div>
            </div>
          )}
        </div>

        <button
          onClick={handleSearch}
          disabled={loading}
          className="btn-primary flex items-center gap-2"
        >
          <Search size={20} />
          {loading ? 'Loading...' : 'Search'}
        </button>
      </div>

      {/* Results */}
      {history && (
        <div className="card">
          <h2 className="text-xl font-semibold text-white mb-4">
            History ({history.history.length} records)
          </h2>
          {history.history.length > 0 ? (
            <div>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="time"
                    stroke="#9ca3af"
                    angle={-45}
                    textAnchor="end"
                    height={100}
                  />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="regime"
                    stroke="#0ea5e9"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
              <div className="mt-4 space-y-2 max-h-64 overflow-y-auto">
                {history.history.map((item, idx) => (
                  <div
                    key={idx}
                    className="bg-slate-700 p-3 rounded-lg flex justify-between items-center"
                  >
                    <span className="text-white">
                      {new Date(item.timestamp).toLocaleString()}
                    </span>
                    <span className="text-primary-400 font-semibold">{item.regime}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-slate-400">No history data found</p>
          )}
        </div>
      )}
    </div>
  )
}

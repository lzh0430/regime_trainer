import { useState, useEffect } from 'react'
import { triggerTraining, listConfigVersions, listAvailableModels, type ConfigVersion } from '../api/services'
import { Play, Loader2 } from 'lucide-react'

interface TrainingTriggerProps {
  onTrainingStarted?: () => void
}

export default function TrainingTrigger({ onTrainingStarted }: TrainingTriggerProps) {
  const [symbol, setSymbol] = useState('BTCUSDT')
  const [timeframe, setTimeframe] = useState('15m')
  const [trainingType, setTrainingType] = useState<'full' | 'incremental'>('full')
  const [configVersionId, setConfigVersionId] = useState<string>('')
  const [configs, setConfigs] = useState<ConfigVersion[]>([])
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  useEffect(() => {
    loadConfigs()
    loadAvailableSymbols()
  }, [])

  const loadConfigs = async () => {
    try {
      const data = await listConfigVersions()
      setConfigs(data.configs.filter((c) => c.is_active))
    } catch (err: any) {
      console.error('Failed to load configs:', err)
    }
  }

  const loadAvailableSymbols = async () => {
    try {
      const data = await listAvailableModels()
      setAvailableSymbols(data.available_models || [])
      // Set default symbol to first available if list is not empty
      if (data.available_models && data.available_models.length > 0) {
        setSymbol(data.available_models[0])
      }
    } catch (err: any) {
      console.error('Failed to load available symbols:', err)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setSuccess(null)
    setLoading(true)

    try {
      const result = await triggerTraining({
        symbol,
        timeframe,
        training_type: trainingType,
        config_version_id: configVersionId || undefined,
      })
      setSuccess(result.message)
      if (onTrainingStarted) {
        onTrainingStarted()
      }
    } catch (err: any) {
      setError(err.message || 'Failed to start training')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-bold mb-4 text-black">Trigger Model Training</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-black mb-1">
            Symbol
          </label>
          <select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-black"
            required
          >
            {availableSymbols.length > 0 ? (
              availableSymbols.map((sym) => (
                <option key={sym} value={sym}>
                  {sym}
                </option>
              ))
            ) : (
              <option value="BTCUSDT">BTCUSDT</option>
            )}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-black mb-1">
            Timeframe
          </label>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-black"
            required
          >
            <option value="5m">5m</option>
            <option value="15m">15m</option>
            <option value="1h">1h</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-black mb-1">
            Training Type
          </label>
          <select
            value={trainingType}
            onChange={(e) => setTrainingType(e.target.value as 'full' | 'incremental')}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-black"
            required
          >
            <option value="full">Full Retrain</option>
            <option value="incremental">Incremental</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-black mb-1">
            Config Version (Optional)
          </label>
          <select
            value={configVersionId}
            onChange={(e) => setConfigVersionId(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-black"
          >
            <option value="">Use defaults (from config.py)</option>
            {configs.map((config) => (
              <option key={config.config_version_id} value={config.config_version_id}>
                {config.config_version_id} {config.description ? `- ${config.description}` : ''}
              </option>
            ))}
          </select>
          <p className="mt-1 text-xs text-black">
            Select a config version from the database, or leave empty to use TrainingConfig defaults
          </p>
        </div>

        {error && (
          <div className="p-3 bg-red-100 border border-red-400 text-red-700 rounded-lg text-sm">
            {error}
          </div>
        )}

        {success && (
          <div className="p-3 bg-green-100 border border-green-400 text-green-700 rounded-lg text-sm">
            {success}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Starting Training...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Start Training
            </>
          )}
        </button>
      </form>
    </div>
  )
}

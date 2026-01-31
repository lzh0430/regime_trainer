import { useState, useEffect } from 'react'
import { RefreshCw, CheckCircle, XCircle } from 'lucide-react'
import {
  listVersions,
  listAvailableModels,
  listModelsByTimeframe,
  getProdVersion,
  setProdVersion,
  type ModelVersion,
} from '../api/services'

export default function Models() {
  const [versions, setVersions] = useState<ModelVersion[]>([])
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [modelsByTimeframe, setModelsByTimeframe] = useState<Record<string, string[]>>({})
  const [loading, setLoading] = useState(true)
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT')
  const [selectedTimeframe, setSelectedTimeframe] = useState('15m')
  const [prodInfo, setProdInfo] = useState<any>(null)

  useEffect(() => {
    fetchData()
  }, [])

  useEffect(() => {
    if (selectedSymbol && selectedTimeframe) {
      fetchProdInfo()
    }
  }, [selectedSymbol, selectedTimeframe])

  const fetchData = async () => {
    setLoading(true)
    try {
      const [versionsData, availableData, timeframeData] = await Promise.all([
        listVersions(),
        listAvailableModels(),
        listModelsByTimeframe(),
      ])
      setVersions(versionsData.versions)
      setAvailableModels(availableData.available_models)
      setModelsByTimeframe(timeframeData)
    } catch (error) {
      console.error('Failed to fetch models:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchProdInfo = async () => {
    try {
      const info = await getProdVersion(selectedSymbol, selectedTimeframe)
      setProdInfo(info)
    } catch (error) {
      console.error('Failed to fetch PROD info:', error)
    }
  }

  const handleSetProd = async (versionId: string) => {
    try {
      await setProdVersion(selectedSymbol, versionId, selectedTimeframe)
      await fetchProdInfo()
      await fetchData()
      alert('PROD version updated successfully')
    } catch (error) {
      console.error('Failed to set PROD:', error)
      alert('Failed to set PROD version')
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    )
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-white">Models</h1>
        <button onClick={fetchData} className="btn-secondary flex items-center gap-2">
          <RefreshCw size={20} />
          Refresh
        </button>
      </div>

      {/* Available Models */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold text-white mb-4">Available Models</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {availableModels.map((model) => (
            <div key={model} className="bg-slate-700 p-3 rounded-lg text-center">
              <p className="text-white">{model}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Models by Timeframe */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold text-white mb-4">Models by Timeframe</h2>
        <div className="space-y-4">
          {Object.entries(modelsByTimeframe).map(([tf, models]) => (
            <div key={tf}>
              <h3 className="text-lg font-medium text-slate-300 mb-2">{tf}</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {models.map((model) => (
                  <div key={model} className="bg-slate-700 p-2 rounded text-sm text-white">
                    {model}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* PROD Management */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold text-white mb-4">PROD Version Management</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Symbol</label>
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="input-field w-full"
            >
              {availableModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Timeframe</label>
            <select
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              className="input-field w-full"
            >
              <option value="5m">5m</option>
              <option value="15m">15m</option>
              <option value="1h">1h</option>
            </select>
          </div>
        </div>
        {prodInfo && (
          <div className="bg-slate-700 p-4 rounded-lg mb-4">
            <p className="text-slate-400 mb-1">Current PROD Version</p>
            <p className="text-xl font-bold text-white">{prodInfo.version_id || 'None'}</p>
            {prodInfo.updated_at && (
              <p className="text-sm text-slate-400 mt-1">
                Updated: {new Date(prodInfo.updated_at).toLocaleString()}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Versions */}
      <div className="card">
        <h2 className="text-xl font-semibold text-white mb-4">All Versions</h2>
        <div className="space-y-4">
          {versions.map((version) => {
            const hasProd = version.contents.some(
              (c) => c.symbol === selectedSymbol && c.timeframe === selectedTimeframe && c.is_prod
            )
            const hasModel = version.contents.some(
              (c) => c.symbol === selectedSymbol && c.timeframe === selectedTimeframe
            )

            return (
              <div key={version.version_id} className="bg-slate-700 p-4 rounded-lg">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="text-lg font-medium text-white">{version.version_id}</h3>
                    <p className="text-sm text-slate-400">
                      Created: {new Date(version.created_at).toLocaleString()}
                    </p>
                  </div>
                  {hasProd && (
                    <span className="flex items-center gap-1 text-green-400">
                      <CheckCircle size={20} />
                      PROD
                    </span>
                  )}
                </div>
                <div className="mt-2">
                  <p className="text-sm text-slate-400 mb-1">Contents:</p>
                  <div className="flex flex-wrap gap-2">
                    {version.contents.map((content, idx) => (
                      <span
                        key={idx}
                        className={`px-2 py-1 rounded text-xs ${
                          content.is_prod
                            ? 'bg-green-600 text-white'
                            : 'bg-slate-600 text-slate-300'
                        }`}
                      >
                        {content.symbol} {content.timeframe}
                      </span>
                    ))}
                  </div>
                </div>
                {hasModel && !hasProd && (
                  <button
                    onClick={() => handleSetProd(version.version_id)}
                    className="btn-primary mt-3 text-sm"
                  >
                    Set as PROD
                  </button>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

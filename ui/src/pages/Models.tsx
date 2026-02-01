import { useState, useEffect } from 'react'
import { RefreshCw, CheckCircle, XCircle, Settings, X } from 'lucide-react'
import { Link } from 'react-router-dom'
import {
  listVersions,
  listAvailableModels,
  listModelsByTimeframe,
  getProdVersion,
  setProdVersion,
  getModelConfig,
  getConfigVersion,
  type ModelVersion,
  type ConfigData,
} from '../api/services'
import TrainingTrigger from '../components/TrainingTrigger'

export default function Models() {
  const [versions, setVersions] = useState<ModelVersion[]>([])
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [modelsByTimeframe, setModelsByTimeframe] = useState<Record<string, string[]>>({})
  const [loading, setLoading] = useState(true)
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT')
  const [selectedTimeframe, setSelectedTimeframe] = useState('15m')
  const [prodInfo, setProdInfo] = useState<any>(null)
  const [configLinks, setConfigLinks] = useState<Record<string, string>>({})
  const [showConfigModal, setShowConfigModal] = useState(false)
  const [selectedConfigVersion, setSelectedConfigVersion] = useState<string | null>(null)
  const [selectedConfigData, setSelectedConfigData] = useState<ConfigData | null>(null)
  const [loadingConfig, setLoadingConfig] = useState(false)

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
      
      // Fetch config links for each version
      const links: Record<string, string> = {}
      for (const version of versionsData.versions) {
        const contents = version.contents || version.symbols || []  // Support both field names
        for (const content of contents) {
          try {
            const configInfo = await getModelConfig(version.version_id, content.symbol, content.timeframe)
            if (configInfo.config_version_id && configInfo.config_version_id !== 'default') {
              const key = `${version.version_id}-${content.symbol}-${content.timeframe}`
              links[key] = configInfo.config_version_id
            }
          } catch (err) {
            // Config not linked, skip
          }
        }
      }
      setConfigLinks(links)
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

  const handleViewConfig = async (configVersionId: string) => {
    try {
      setLoadingConfig(true)
      setSelectedConfigVersion(configVersionId)
      const configData = await getConfigVersion(configVersionId)
      setSelectedConfigData(configData)
      setShowConfigModal(true)
    } catch (error) {
      console.error('Failed to load config:', error)
      alert('Failed to load config details')
    } finally {
      setLoadingConfig(false)
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

      {/* Model Training */}
      <div className="card mb-6">
        <TrainingTrigger onTrainingStarted={fetchData} />
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
            const contents = version.contents || version.symbols || []  // Support both field names
            const hasProd = contents.some(
              (c) => c.symbol === selectedSymbol && c.timeframe === selectedTimeframe && c.is_prod
            )
            const hasModel = contents.some(
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
                  <div className="space-y-2">
                    {contents.map((content, idx) => {
                      const configKey = `${version.version_id}-${content.symbol}-${content.timeframe}`
                      const configVersionId = configLinks[configKey]
                      return (
                        <div key={idx} className="flex items-center gap-2 flex-wrap">
                          <span
                            className={`px-2 py-1 rounded text-xs ${
                              content.is_prod
                                ? 'bg-green-600 text-white'
                                : 'bg-slate-600 text-slate-300'
                            }`}
                          >
                            {content.symbol} {content.timeframe}
                          </span>
                          {configVersionId ? (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.preventDefault()
                                e.stopPropagation()
                                handleViewConfig(configVersionId)
                              }}
                              className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-blue-600 text-white hover:bg-blue-700 cursor-pointer transition-colors"
                              title={`Click to view config: ${configVersionId}`}
                            >
                              <Settings size={12} />
                              Config: {configVersionId}
                            </button>
                          ) : (
                            <span className="px-2 py-1 rounded text-xs bg-gray-600 text-gray-300" title="Using default config">
                              Default Config
                            </span>
                          )}
                        </div>
                      )
                    })}
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

      {/* Config Detail Modal */}
      {showConfigModal && selectedConfigVersion && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => {
          setShowConfigModal(false)
          setSelectedConfigVersion(null)
          setSelectedConfigData(null)
        }}>
          <div className="bg-white rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-black">Config: {selectedConfigVersion}</h2>
              <button
                type="button"
                onClick={() => {
                  setShowConfigModal(false)
                  setSelectedConfigVersion(null)
                  setSelectedConfigData(null)
                }}
                className="text-gray-600 hover:text-gray-800"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            {loadingConfig ? (
              <div className="text-center py-8 text-black">Loading config...</div>
            ) : selectedConfigData ? (
              <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                <pre className="text-xs overflow-x-auto text-black font-mono">
                  {JSON.stringify(selectedConfigData, null, 2)}
                </pre>
              </div>
            ) : (
              <div className="text-center py-8 text-black">Failed to load config data</div>
            )}
            <div className="mt-4 flex gap-2 justify-end">
              <Link
                to={`/configs`}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Go to Config Management
              </Link>
              <button
                onClick={() => {
                  setShowConfigModal(false)
                  setSelectedConfigVersion(null)
                  setSelectedConfigData(null)
                }}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 text-black"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

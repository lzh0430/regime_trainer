import { useState, useEffect } from 'react'
import {
  listConfigVersions,
  getConfigVersion,
  initConfigFromFile,
  getDefaultConfig,
  createConfigVersion,
  updateConfigVersion,
  deleteConfigVersion,
  getModelsForConfig,
  type ConfigVersion,
  type ConfigData,
} from '../api/services'
import { Plus, Trash2, Edit, Eye, FileText, Copy, Check, X, ChevronDown, ChevronRight } from 'lucide-react'

export default function ConfigManagement() {
  const [configs, setConfigs] = useState<ConfigVersion[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedConfig, setSelectedConfig] = useState<ConfigVersion | null>(null)
  const [configData, setConfigData] = useState<ConfigData | null>(null)
  const [editingConfig, setEditingConfig] = useState<ConfigData | null>(null)
  const [editDescription, setEditDescription] = useState('')
  const [editMode, setEditMode] = useState<'form' | 'json'>('form')
  const [jsonEditorValue, setJsonEditorValue] = useState<string>('')
  const [jsonError, setJsonError] = useState<string | null>(null)
  const [expandedConfigs, setExpandedConfigs] = useState<Set<string>>(new Set())
  const [configDetailsCache, setConfigDetailsCache] = useState<Record<string, ConfigData>>({})
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showEditModal, setShowEditModal] = useState(false)
  const [showViewModal, setShowViewModal] = useState(false)
  const [showInitModal, setShowInitModal] = useState(false)
  const [initDescription, setInitDescription] = useState('Initial config from TrainingConfig')
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  useEffect(() => {
    loadConfigs()
  }, [])

  const loadConfigs = async () => {
    try {
      setLoading(true)
      const data = await listConfigVersions()
      setConfigs(data.configs)
      if (data.configs.length === 0) {
        setShowInitModal(true)
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load configs')
    } finally {
      setLoading(false)
    }
  }

  const handleInit = async () => {
    try {
      setError(null)
      await initConfigFromFile(initDescription)
      setSuccess('Config initialized successfully')
      setShowInitModal(false)
      await loadConfigs()
    } catch (err: any) {
      setError(err.message || 'Failed to initialize config')
    }
  }

  const handleView = async (config: ConfigVersion) => {
    try {
      setError(null)
      const data = await getConfigVersion(config.config_version_id)
      setConfigData(data)
      setSelectedConfig(config)
      setShowViewModal(true)
    } catch (err: any) {
      setError(err.message || 'Failed to load config data')
    }
  }

  const toggleExpand = async (configVersionId: string) => {
    const newExpanded = new Set(expandedConfigs)
    if (newExpanded.has(configVersionId)) {
      newExpanded.delete(configVersionId)
    } else {
      newExpanded.add(configVersionId)
      // Load config details if not cached
      if (!configDetailsCache[configVersionId]) {
        try {
          const data = await getConfigVersion(configVersionId)
          setConfigDetailsCache({ ...configDetailsCache, [configVersionId]: data })
        } catch (err: any) {
          console.error('Failed to load config details:', err)
        }
      }
    }
    setExpandedConfigs(newExpanded)
  }

  const organizeConfigData = (data: ConfigData): Record<string, any> => {
    const organized: Record<string, any> = {
      'Basic Settings': {},
      'Model Configs': {},
      'Training Settings': {},
      'HMM Settings': {},
      'Data Settings': {},
      'Other': {},
    }

    for (const [key, value] of Object.entries(data)) {
      if (key.startsWith('MODEL_CONFIGS.')) {
        const parts = key.split('.')
        if (parts.length >= 3) {
          const timeframe = parts[1]
          const param = parts[2]
          if (!organized['Model Configs'][timeframe]) {
            organized['Model Configs'][timeframe] = {}
          }
          organized['Model Configs'][timeframe][param] = value
        } else {
          organized['Other'][key] = value
        }
      } else if (['SYMBOLS', 'TIMEFRAMES', 'PRIMARY_TIMEFRAME', 'ENABLED_MODELS'].includes(key)) {
        organized['Basic Settings'][key] = value
      } else if (key.includes('TRAIN') || key.includes('EPOCH') || key.includes('BATCH') || key.includes('LEARNING')) {
        organized['Training Settings'][key] = value
      } else if (key.includes('HMM') || key.includes('N_STATES') || key.includes('REGIME')) {
        organized['HMM Settings'][key] = value
      } else if (key.includes('DATA') || key.includes('CACHE') || key.includes('DIR')) {
        organized['Data Settings'][key] = value
      } else {
        organized['Other'][key] = value
      }
    }

    // Remove empty categories
    Object.keys(organized).forEach((key) => {
      if (Object.keys(organized[key]).length === 0) {
        delete organized[key]
      }
    })

    return organized
  }

  const handleDelete = async (configVersionId: string) => {
    if (!confirm(`Delete config version ${configVersionId}?`)) return

    try {
      setError(null)
      await deleteConfigVersion(configVersionId)
      setSuccess('Config deleted successfully')
      await loadConfigs()
    } catch (err: any) {
      setError(err.message || 'Failed to delete config')
    }
  }

  const handleCreateFromDefault = async () => {
    try {
      setError(null)
      const defaultConfig = await getDefaultConfig()
      setEditingConfig(defaultConfig)
      setJsonEditorValue(JSON.stringify(defaultConfig, null, 2))
      setEditDescription('Config created from defaults')
      setEditMode('form')
      setJsonError(null)
      setShowCreateModal(false)
      setShowEditModal(true)
    } catch (err: any) {
      setError(err.message || 'Failed to load default config')
    }
  }

  const handleClone = async (config: ConfigVersion) => {
    try {
      setError(null)
      const data = await getConfigVersion(config.config_version_id)
      setEditingConfig(data)
      setJsonEditorValue(JSON.stringify(data, null, 2))
      setEditDescription(`Cloned from ${config.config_version_id}`)
      setEditMode('form')
      setJsonError(null)
      setShowEditModal(true)
    } catch (err: any) {
      setError(err.message || 'Failed to load config for cloning')
    }
  }

  const handleEditSubmit = async () => {
    if (!editingConfig) return

    try {
      setError(null)
      setJsonError(null)
      
      // If in JSON mode, parse and validate JSON first
      let configToSubmit = editingConfig
      if (editMode === 'json') {
        try {
          const parsed = JSON.parse(jsonEditorValue)
          if (typeof parsed !== 'object' || Array.isArray(parsed)) {
            setJsonError('Config must be a JSON object')
            return
          }
          configToSubmit = parsed
        } catch (err: any) {
          setJsonError(`Invalid JSON: ${err.message}`)
          return
        }
      }
      
      await createConfigVersion(configToSubmit, editDescription || 'Modified config')
      setSuccess('Config created successfully')
      setShowEditModal(false)
      setEditingConfig(null)
      setEditDescription('')
      setJsonEditorValue('')
      setEditMode('form')
      setJsonError(null)
      await loadConfigs()
    } catch (err: any) {
      setError(err.message || 'Failed to create config')
    }
  }

  const handleEditValueChange = (key: string, value: any) => {
    if (!editingConfig) return
    const updated = { ...editingConfig, [key]: value }
    setEditingConfig(updated)
    // Also update JSON editor if in JSON mode
    if (editMode === 'json') {
      setJsonEditorValue(JSON.stringify(updated, null, 2))
    }
  }

  const handleJsonEditorChange = (value: string) => {
    setJsonEditorValue(value)
    setJsonError(null)
    // Try to parse and update editingConfig in real-time for validation
    try {
      const parsed = JSON.parse(value)
      if (typeof parsed === 'object' && !Array.isArray(parsed)) {
        setEditingConfig(parsed)
      }
    } catch {
      // Invalid JSON, but don't show error until submit
    }
  }

  const handleModeSwitch = (mode: 'form' | 'json') => {
    if (mode === 'json' && editingConfig) {
      // Switch to JSON mode - serialize current config
      setJsonEditorValue(JSON.stringify(editingConfig, null, 2))
      setJsonError(null)
    } else if (mode === 'form' && jsonEditorValue) {
      // Switch to form mode - parse JSON and update config
      try {
        const parsed = JSON.parse(jsonEditorValue)
        if (typeof parsed === 'object' && !Array.isArray(parsed)) {
          setEditingConfig(parsed)
          setJsonError(null)
        } else {
          setJsonError('Config must be a JSON object')
          return
        }
      } catch (err: any) {
        setJsonError(`Invalid JSON: ${err.message}`)
        return
      }
    }
    setEditMode(mode)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading configs...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Config Management</h1>
        <div className="flex gap-2">
          {configs.length === 0 && (
            <button
              onClick={() => setShowInitModal(true)}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
            >
              <FileText className="w-4 h-4" />
              Initialize from File
            </button>
          )}
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Create Config
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-900 border border-red-600 text-red-200 rounded-lg">
          {error}
        </div>
      )}

      {success && (
        <div className="p-4 bg-green-900 border border-green-600 text-green-200 rounded-lg">
          {success}
        </div>
      )}

      {configs.length === 0 ? (
        <div className="text-center py-12 bg-slate-800 rounded-lg">
          <FileText className="w-16 h-16 mx-auto text-slate-400 mb-4" />
          <h3 className="text-xl font-semibold mb-2 text-white">No Configs Found</h3>
          <p className="text-slate-400 mb-4">
            Initialize from TrainingConfig file to get started
          </p>
          <button
            onClick={() => setShowInitModal(true)}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Initialize from File
          </button>
        </div>
      ) : (
        <div className="card overflow-hidden">
          <table className="min-w-full divide-y divide-slate-700">
            <thead className="bg-slate-800">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Version ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Created At
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Description
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Models
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-slate-800 divide-y divide-slate-700">
              {configs.map((config) => {
                const isExpanded = expandedConfigs.has(config.config_version_id)
                const details = configDetailsCache[config.config_version_id]
                const organizedDetails = details ? organizeConfigData(details) : null

                return (
                  <>
                    <tr key={config.config_version_id} className="hover:bg-slate-700">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-white">
                        <button
                          onClick={() => toggleExpand(config.config_version_id)}
                          className="flex items-center gap-2 hover:text-blue-400 text-white"
                        >
                          {isExpanded ? (
                            <ChevronDown className="w-4 h-4" />
                          ) : (
                            <ChevronRight className="w-4 h-4" />
                          )}
                          {config.config_version_id}
                        </button>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">
                        {new Date(config.created_at).toLocaleString()}
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-300">
                        {config.description || '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">
                        {config.model_count}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {config.is_active ? (
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-600 text-white">
                            Active
                          </span>
                        ) : (
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-600 text-white">
                            Deleted
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <div className="flex justify-end gap-2">
                          <button
                            onClick={() => handleView(config)}
                            className="text-blue-400 hover:text-blue-300"
                            title="View Full Details"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleClone(config)}
                            className="text-green-400 hover:text-green-300"
                            title="Clone"
                          >
                            <Copy className="w-4 h-4" />
                          </button>
                          {config.is_active && (
                            <button
                              onClick={() => handleDelete(config.config_version_id)}
                              className="text-red-400 hover:text-red-300"
                              title="Delete"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                    {isExpanded && organizedDetails && (
                      <tr>
                        <td colSpan={6} className="px-6 py-4 bg-slate-900">
                          <div className="space-y-4">
                            <h4 className="font-semibold text-white">Config Details</h4>
                            {Object.entries(organizedDetails).map(([category, values]) => (
                              <div key={category} className="border-l-2 border-blue-500 pl-4">
                                <h5 className="font-medium text-slate-300 mb-2">{category}</h5>
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 text-sm">
                                  {Object.entries(values).map(([key, value]) => (
                                    <div key={key} className="bg-slate-800 p-2 rounded border border-slate-700">
                                      <span className="font-mono text-xs text-slate-400">{key}:</span>
                                      <span className="ml-2 text-slate-200 break-all">
                                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Init Modal */}
      {showInitModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h2 className="text-xl font-bold mb-4">Initialize Config from File</h2>
            <p className="text-gray-600 mb-4">
              Import current TrainingConfig values into the database as the first version.
            </p>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description
              </label>
              <input
                type="text"
                value={initDescription}
                onChange={(e) => setInitDescription(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                placeholder="Initial config from TrainingConfig"
              />
            </div>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setShowInitModal(false)}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleInit}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Initialize
              </button>
            </div>
          </div>
        </div>
      )}

      {/* View Modal */}
      {showViewModal && selectedConfig && configData && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-black">Config: {selectedConfig.config_version_id}</h2>
              <button
                onClick={() => {
                  setShowViewModal(false)
                  setSelectedConfig(null)
                  setConfigData(null)
                }}
                className="text-gray-600 hover:text-gray-800"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="mb-4">
              <p className="text-sm text-black">
                Created: {new Date(selectedConfig.created_at).toLocaleString()}
              </p>
              {selectedConfig.description && (
                <p className="text-sm text-black">Description: {selectedConfig.description}</p>
              )}
            </div>
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <pre className="text-xs overflow-x-auto text-black font-mono">
                {JSON.stringify(configData, null, 2)}
              </pre>
            </div>
            <div className="mt-4 flex gap-2 justify-end">
              <button
                onClick={() => {
                  setShowViewModal(false)
                  setSelectedConfig(null)
                  setConfigData(null)
                }}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 text-black"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h2 className="text-xl font-bold mb-4 text-black">Create Config</h2>
            <p className="text-gray-600 mb-4">
              Create a new config version. You can start from defaults or clone an existing config.
            </p>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 text-black"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateFromDefault}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Create from Defaults
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Edit/Clone Modal */}
      {showEditModal && editingConfig && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-6xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-black">Edit Config</h2>
              <button
                onClick={() => {
                  setShowEditModal(false)
                  setEditingConfig(null)
                  setEditDescription('')
                }}
                className="text-gray-600 hover:text-gray-800"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-black mb-2">
                Description
              </label>
              <input
                type="text"
                value={editDescription}
                onChange={(e) => setEditDescription(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-black"
                placeholder="Description for this config version"
              />
            </div>

            {/* Mode Toggle */}
            <div className="mb-4 flex gap-2 border-b border-gray-300">
              <button
                type="button"
                onClick={() => handleModeSwitch('form')}
                className={`px-4 py-2 font-medium text-sm ${
                  editMode === 'form'
                    ? 'text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                Form View
              </button>
              <button
                type="button"
                onClick={() => handleModeSwitch('json')}
                className={`px-4 py-2 font-medium text-sm ${
                  editMode === 'json'
                    ? 'text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                JSON Editor
              </button>
            </div>

            {editMode === 'json' ? (
              <div className="space-y-2">
                <label className="block text-sm font-medium text-black">
                  Config JSON
                </label>
                <textarea
                  value={jsonEditorValue}
                  onChange={(e) => handleJsonEditorChange(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-black text-sm font-mono"
                  rows={20}
                  placeholder='{"SYMBOLS": ["BTCUSDT"], ...}'
                />
                {jsonError && (
                  <div className="p-2 bg-red-100 border border-red-400 text-red-700 rounded text-sm">
                    {jsonError}
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4 max-h-[60vh] overflow-y-auto">
              {Object.entries(organizeConfigData(editingConfig)).map(([category, values]) => (
                <div key={category} className="border-l-2 border-blue-500 pl-4">
                  <h5 className="font-medium text-black mb-2">{category}</h5>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {Object.entries(values).map(([key, value]) => {
                      const currentValue = editingConfig[key] || value
                      const isString = typeof currentValue === 'string'
                      const isNumber = typeof currentValue === 'number'
                      const isBoolean = typeof currentValue === 'boolean'
                      const isArray = Array.isArray(currentValue)
                      const isObject = typeof currentValue === 'object' && !isArray

                      return (
                        <div key={key} className="bg-gray-50 p-3 rounded border border-gray-200">
                          <label className="block text-xs font-mono text-gray-600 mb-1">
                            {key}
                          </label>
                          {isBoolean ? (
                            <select
                              value={String(currentValue)}
                              onChange={(e) => handleEditValueChange(key, e.target.value === 'true')}
                              className="w-full px-2 py-1 border border-gray-300 rounded text-black text-sm"
                            >
                              <option value="true">true</option>
                              <option value="false">false</option>
                            </select>
                          ) : isNumber ? (
                            <input
                              type="number"
                              value={currentValue}
                              onChange={(e) => {
                                const numValue = e.target.value.includes('.')
                                  ? parseFloat(e.target.value)
                                  : parseInt(e.target.value, 10)
                                handleEditValueChange(key, isNaN(numValue) ? 0 : numValue)
                              }}
                              className="w-full px-2 py-1 border border-gray-300 rounded text-black text-sm"
                            />
                          ) : isArray ? (
                            <textarea
                              value={JSON.stringify(currentValue)}
                              onChange={(e) => {
                                try {
                                  const parsed = JSON.parse(e.target.value)
                                  if (Array.isArray(parsed)) {
                                    handleEditValueChange(key, parsed)
                                  }
                                } catch {
                                  // Invalid JSON, keep as is
                                }
                              }}
                              className="w-full px-2 py-1 border border-gray-300 rounded text-black text-sm font-mono"
                              rows={2}
                              placeholder="JSON array"
                            />
                          ) : isObject ? (
                            <textarea
                              value={JSON.stringify(currentValue, null, 2)}
                              onChange={(e) => {
                                try {
                                  const parsed = JSON.parse(e.target.value)
                                  handleEditValueChange(key, parsed)
                                } catch {
                                  // Invalid JSON, keep as is
                                }
                              }}
                              className="w-full px-2 py-1 border border-gray-300 rounded text-black text-sm font-mono"
                              rows={3}
                              placeholder="JSON object"
                            />
                          ) : (
                            <input
                              type="text"
                              value={String(currentValue)}
                              onChange={(e) => handleEditValueChange(key, e.target.value)}
                              className="w-full px-2 py-1 border border-gray-300 rounded text-black text-sm"
                            />
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              ))}
              </div>
            )}

            <div className="mt-6 flex gap-2 justify-end">
              <button
                onClick={() => {
                  setShowEditModal(false)
                  setEditingConfig(null)
                  setEditDescription('')
                }}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 text-black"
              >
                Cancel
              </button>
              <button
                onClick={handleEditSubmit}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                Create New Version
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

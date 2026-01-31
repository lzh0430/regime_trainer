import apiClient from './client'

// Types
export interface PredictionResult {
  symbol: string
  timeframe: string
  most_likely_regime: {
    name: string
    probability: number
  }
  all_regimes: Array<{
    name: string
    probability: number
  }>
}

export interface MultiStepPrediction {
  symbol: string
  timeframe: string
  predictions: Array<{
    step: number
    regimes: Array<{
      name: string
      probability: number
    }>
  }>
  history?: Array<{
    timestamp: string
    regime: string
  }>
}

export interface ModelMetadata {
  symbol: string
  timeframe: string
  regime_mapping: Record<string, number>
  model_config: Record<string, any>
}

export interface ModelVersion {
  version_id: string
  created_at: string
  contents: Array<{
    symbol: string
    timeframe: string
    is_prod: boolean
  }>
}

export interface ForwardTestStatus {
  is_running: boolean
  thread_alive: boolean
  registered_jobs: Array<{
    version_id: string
    symbol: string
    timeframe: string
    interval_minutes: number
    next_run: string
  }>
  total_jobs: number
}

export interface ForwardTestResult {
  total_campaigns: number
  successful_runs: number
  failed_runs: number
  skipped_runs: number
  results: Array<any>
}

export interface HistoryData {
  symbol: string
  timeframe: string
  history: Array<{
    timestamp: string
    regime: string
  }>
}

// API Services
export const healthCheck = async () => {
  const response = await apiClient.get('/health')
  return response.data
}

export const predictNextRegime = async (symbol: string, timeframe: string = '15m') => {
  const response = await apiClient.get<PredictionResult>(`/predict/${symbol}`, {
    params: { timeframe },
  })
  return response.data
}

export const predictRegimes = async (
  symbol: string,
  timeframe: string = '15m',
  includeHistory: boolean = true
) => {
  const response = await apiClient.get<MultiStepPrediction>(`/predict_regimes/${symbol}`, {
    params: { timeframe, include_history: includeHistory },
  })
  return response.data
}

export const batchPredict = async (symbols: string[], timeframe: string = '15m') => {
  const response = await apiClient.post<Record<string, PredictionResult>>('/batch_predict', {
    symbols,
    timeframe,
  })
  return response.data
}

export const getModelMetadata = async (symbol: string, timeframe: string = '15m') => {
  const response = await apiClient.get<ModelMetadata>(`/metadata/${symbol}`, {
    params: { timeframe },
  })
  return response.data
}

export const listAvailableModels = async (timeframe?: string) => {
  const params = timeframe ? { timeframe } : {}
  const response = await apiClient.get<{ available_models: string[]; count: number }>(
    '/models/available',
    { params }
  )
  return response.data
}

export const listModelsByTimeframe = async () => {
  const response = await apiClient.get<Record<string, string[]>>('/models/by_timeframe')
  return response.data
}

export const listVersions = async () => {
  const response = await apiClient.get<{ versions: ModelVersion[] }>('/models/versions')
  return response.data
}

export const getProdVersion = async (symbol: string, timeframe: string = '15m') => {
  const response = await apiClient.get('/models/prod', {
    params: { symbol, timeframe },
  })
  return response.data
}

export const setProdVersion = async (
  symbol: string,
  versionId: string,
  timeframe: string = '15m'
) => {
  const response = await apiClient.post('/models/prod', {
    symbol,
    version_id: versionId,
    timeframe,
  })
  return response.data
}

export const getForwardTestStatus = async () => {
  const response = await apiClient.get<ForwardTestStatus>('/forward_test/status')
  return response.data
}

export const triggerAllForwardTests = async () => {
  const response = await apiClient.post<ForwardTestResult>('/forward_test/trigger_all')
  return response.data
}

export const getHistory = async (
  symbol: string,
  timeframe: string = '15m',
  options?: {
    lookback_hours?: number
    start_date?: string
    end_date?: string
  }
) => {
  const response = await apiClient.get<HistoryData>(`/history/${symbol}`, {
    params: { timeframe, ...options },
  })
  return response.data
}

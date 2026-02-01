import { useState, useEffect } from 'react'
import { RefreshCw, Play, CheckCircle, XCircle, Clock } from 'lucide-react'
import {
  getForwardTestStatus,
  triggerAllForwardTests,
  type ForwardTestStatus,
  type ForwardTestResult,
} from '../api/services'

export default function ForwardTesting() {
  const [status, setStatus] = useState<ForwardTestStatus | null>(null)
  const [triggerResult, setTriggerResult] = useState<ForwardTestResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [triggering, setTriggering] = useState(false)

  useEffect(() => {
    fetchStatus()
    // Refresh every 10 seconds
    const interval = setInterval(fetchStatus, 10000)
    return () => clearInterval(interval)
  }, [])

  const fetchStatus = async () => {
    try {
      const data = await getForwardTestStatus()
      setStatus(data)
    } catch (error) {
      console.error('Failed to fetch forward test status:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleTriggerAll = async () => {
    setTriggering(true)
    try {
      const result = await triggerAllForwardTests()
      setTriggerResult(result)
      await fetchStatus()
    } catch (error) {
      console.error('Failed to trigger forward tests:', error)
      alert('Failed to trigger forward tests')
    } finally {
      setTriggering(false)
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
        <h1 className="text-3xl font-bold text-white">Forward Testing</h1>
        <button onClick={fetchStatus} className="btn-secondary flex items-center gap-2">
          <RefreshCw size={20} />
          Refresh
        </button>
      </div>

      {/* Status Card */}
      {status && (
        <div className="card mb-6">
          <h2 className="text-xl font-semibold text-white mb-4">Cron Manager Status</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-700 p-4 rounded-lg">
              <p className="text-slate-400 text-sm mb-1">Scheduler Running</p>
              <div className="flex items-center gap-2">
                {status.is_running ? (
                  <>
                    <CheckCircle className="text-green-400" size={20} />
                    <span className="text-white font-semibold">Yes</span>
                  </>
                ) : (
                  <>
                    <XCircle className="text-red-400" size={20} />
                    <span className="text-white font-semibold">No</span>
                  </>
                )}
              </div>
            </div>
            <div className="bg-slate-700 p-4 rounded-lg">
              <p className="text-slate-400 text-sm mb-1">Thread Alive</p>
              <div className="flex items-center gap-2">
                {status.thread_alive ? (
                  <>
                    <CheckCircle className="text-green-400" size={20} />
                    <span className="text-white font-semibold">Yes</span>
                  </>
                ) : (
                  <>
                    <XCircle className="text-red-400" size={20} />
                    <span className="text-white font-semibold">No</span>
                  </>
                )}
              </div>
            </div>
            <div className="bg-slate-700 p-4 rounded-lg">
              <p className="text-slate-400 text-sm mb-1">Total Jobs</p>
              <p className="text-2xl font-bold text-white">{status.total_jobs}</p>
            </div>
          </div>
        </div>
      )}

      {/* Registered Jobs */}
      {status && status.registered_jobs.length > 0 && (
        <div className="card mb-6">
          <h2 className="text-xl font-semibold text-white mb-4">Registered Cron Jobs</h2>
          <div className="space-y-3">
            {status.registered_jobs.map((job, idx) => (
              <div key={idx} className="bg-slate-700 p-4 rounded-lg">
                <div className="flex justify-between items-start">
                  <div>
                    <p className="text-white font-medium">
                      {job.version_id} - {job.symbol} ({job.timeframe})
                    </p>
                    <p className="text-sm text-slate-400">
                      Interval: {job.interval_minutes} minutes
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-slate-400">Next Run</p>
                    <p className="text-white text-sm">
                      {job.next_run !== 'unknown' ? new Date(job.next_run).toLocaleString() : 'Unknown'}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Trigger All */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold text-white mb-4">Manual Trigger</h2>
        <button
          onClick={handleTriggerAll}
          disabled={triggering}
          className="btn-primary flex items-center gap-2"
        >
          <Play size={20} />
          {triggering ? 'Triggering...' : 'Trigger All Pending Tests'}
        </button>
      </div>

      {/* Trigger Results */}
      {triggerResult && (
        <div className="card">
          <h2 className="text-xl font-semibold text-white mb-4">Last Trigger Results</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-slate-700 p-4 rounded-lg">
              <p className="text-slate-400 text-sm mb-1">Total Campaigns</p>
              <p className="text-2xl font-bold text-white">{triggerResult.total_campaigns}</p>
            </div>
            <div className="bg-slate-700 p-4 rounded-lg">
              <p className="text-slate-400 text-sm mb-1">Successful</p>
              <p className="text-2xl font-bold text-green-400">{triggerResult.successful_runs}</p>
            </div>
            <div className="bg-slate-700 p-4 rounded-lg">
              <p className="text-slate-400 text-sm mb-1">Failed</p>
              <p className="text-2xl font-bold text-red-400">{triggerResult.failed_runs}</p>
            </div>
            <div className="bg-slate-700 p-4 rounded-lg">
              <p className="text-slate-400 text-sm mb-1">Skipped</p>
              <p className="text-2xl font-bold text-yellow-400">{triggerResult.skipped_runs}</p>
            </div>
          </div>
          {triggerResult.results.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-lg font-medium text-white mb-2">Results</h3>
              {triggerResult.results.map((result: any, idx: number) => (
                <div key={idx} className="bg-slate-700 p-3 rounded-lg">
                  <div className="flex justify-between items-center">
                    <div>
                      <p className="text-white">
                        {result.symbol} ({result.timeframe}) - {result.version_id}
                      </p>
                      {result.status === 'success' && result.runs_count && (
                        <p className="text-sm text-slate-400">
                          Runs: {result.runs_count}/{result.required_runs}
                        </p>
                      )}
                    </div>
                    <span
                      className={`px-3 py-1 rounded text-sm ${
                        result.status === 'success'
                          ? 'bg-green-600 text-white'
                          : result.status === 'error'
                          ? 'bg-red-600 text-white'
                          : 'bg-yellow-600 text-white'
                      }`}
                    >
                      {result.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

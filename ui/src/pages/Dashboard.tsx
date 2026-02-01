import { useEffect, useState } from 'react'
import { Activity, TrendingUp, Database, TestTube } from 'lucide-react'
import { healthCheck, listAvailableModels, getForwardTestStatus } from '../api/services'

export default function Dashboard() {
  const [health, setHealth] = useState<any>(null)
  const [models, setModels] = useState<{ count: number } | null>(null)
  const [forwardTest, setForwardTest] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [healthData, modelsData, forwardTestData] = await Promise.all([
          healthCheck(),
          listAvailableModels(),
          getForwardTestStatus().catch(() => null),
        ])
        setHealth(healthData)
        setModels(modelsData)
        setForwardTest(forwardTestData)
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    )
  }

  const stats = [
    {
      name: 'API Status',
      value: health?.status || 'Unknown',
      icon: Activity,
      color: health?.status === 'healthy' ? 'text-green-400' : 'text-red-400',
    },
    {
      name: 'Available Models',
      value: models?.count || 0,
      icon: Database,
      color: 'text-blue-400',
    },
    {
      name: 'Active Cron Jobs',
      value: forwardTest?.total_jobs || 0,
      icon: TestTube,
      color: 'text-purple-400',
    },
    {
      name: 'Scheduler Status',
      value: forwardTest?.is_running ? 'Running' : 'Stopped',
      icon: TrendingUp,
      color: forwardTest?.is_running ? 'text-green-400' : 'text-yellow-400',
    },
  ]

  return (
    <div>
      <h1 className="text-3xl font-bold text-white mb-8">Dashboard</h1>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat) => {
          const Icon = stat.icon
          return (
            <div key={stat.name} className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-400 text-sm mb-1">{stat.name}</p>
                  <p className={`text-2xl font-bold ${stat.color}`}>{stat.value}</p>
                </div>
                <Icon size={32} className={stat.color} />
              </div>
            </div>
          )
        })}
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h2 className="text-xl font-semibold text-white mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <a
            href="/predictions"
            className="p-4 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
          >
            <h3 className="font-medium text-white mb-1">Make Prediction</h3>
            <p className="text-sm text-slate-400">Predict market regime for a symbol</p>
          </a>
          <a
            href="/models"
            className="p-4 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
          >
            <h3 className="font-medium text-white mb-1">Manage Models</h3>
            <p className="text-sm text-slate-400">View and manage model versions</p>
          </a>
          <a
            href="/forward-testing"
            className="p-4 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
          >
            <h3 className="font-medium text-white mb-1">Forward Testing</h3>
            <p className="text-sm text-slate-400">Monitor forward test campaigns</p>
          </a>
        </div>
      </div>

      {/* System Info */}
      {health && (
        <div className="card mt-6">
          <h2 className="text-xl font-semibold text-white mb-4">System Information</h2>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-slate-400">API Status:</span>
              <span className="text-white">{health.status}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Last Check:</span>
              <span className="text-white">
                {new Date(health.timestamp).toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

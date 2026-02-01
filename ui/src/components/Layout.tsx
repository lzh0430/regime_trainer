import { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { 
  LayoutDashboard, 
  TrendingUp, 
  Database, 
  TestTube, 
  History as HistoryIcon,
  Settings
} from 'lucide-react'

interface LayoutProps {
  children: ReactNode
}

const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/configs', label: 'Config Management', icon: Settings },
  { path: '/models', label: 'Model Management', icon: Database },
  { path: '/forward-testing', label: 'Forward Testing', icon: TestTube },
  { path: '/predictions', label: 'Predictions', icon: TrendingUp },
  { path: '/history', label: 'History', icon: HistoryIcon },
]

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 h-full w-64 bg-slate-800 border-r border-slate-700">
        <div className="p-6">
          <h1 className="text-2xl font-bold text-white mb-2">Regime Trainer</h1>
          <p className="text-sm text-slate-400">Market Regime Prediction</p>
        </div>
        <nav className="px-4 space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                  isActive
                    ? 'bg-primary-600 text-white'
                    : 'text-slate-300 hover:bg-slate-700'
                }`}
              >
                <Icon size={20} />
                <span>{item.label}</span>
              </Link>
            )
          })}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="ml-64 p-8">
        {children}
      </main>
    </div>
  )
}

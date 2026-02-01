import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Predictions from './pages/Predictions'
import Models from './pages/Models'
import ForwardTesting from './pages/ForwardTesting'
import History from './pages/History'
import ConfigManagement from './pages/ConfigManagement'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/models" element={<Models />} />
          <Route path="/forward-testing" element={<ForwardTesting />} />
          <Route path="/history" element={<History />} />
          <Route path="/configs" element={<ConfigManagement />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App

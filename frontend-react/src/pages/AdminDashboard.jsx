import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import './AdminDashboard.css'

function AdminDashboard() {
  const navigate = useNavigate()
  const [stats, setStats] = useState({
    totalUsers: 0,
    totalPredictions: 0,
    recentPredictions: []
  })
  const [predictionsByUser, setPredictionsByUser] = useState({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const token = localStorage.getItem('token')
        
        // Fetch users from backend
        const usersRes = await axios.get('http://localhost:8000/admin/users', {
          headers: { 'Authorization': `Bearer ${token}` }
        })
        
        // Load predictions from localStorage (grouped by user)
        const history = JSON.parse(localStorage.getItem('history') || '[]')
        
        // Group predictions by username
        const grouped = {}
        history.forEach(pred => {
          const username = pred.username || 'Unknown'
          if (!grouped[username]) {
            grouped[username] = []
          }
          grouped[username].push(pred)
        })
        
        setPredictionsByUser(grouped)
        setStats({
          totalUsers: usersRes.data.total || 0,
          totalPredictions: history.length,
          recentPredictions: history.slice(0, 10)
        })
      } catch (error) {
        console.error('Error fetching admin data:', error)
        // Fallback to localStorage only
        const history = JSON.parse(localStorage.getItem('history') || '[]')
        const grouped = {}
        history.forEach(pred => {
          const username = pred.username || 'Unknown'
          if (!grouped[username]) {
            grouped[username] = []
          }
          grouped[username].push(pred)
        })
        setPredictionsByUser(grouped)
        setStats({
          totalUsers: 0,
          totalPredictions: history.length,
          recentPredictions: history.slice(0, 10)
        })
      } finally {
        setLoading(false)
      }
    }
    
    fetchData()
  }, [])

  const formatDate = (timestamp) => {
    if (!timestamp) return 'N/A'
    const date = new Date(timestamp)
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    })
  }

  return (
    <div className="admin-dashboard">
      <div className="admin-container">
        <div className="admin-header">
          <h1>Admin Dashboard</h1>
          <p>Manage and monitor the LeafScan application</p>
        </div>

        {/* Stats Cards */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon">👥</div>
            <div className="stat-content">
              <h3>Total Users</h3>
              <p className="stat-value">{stats.totalUsers}</p>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">🔍</div>
            <div className="stat-content">
              <h3>Total Predictions</h3>
              <p className="stat-value">{stats.totalPredictions}</p>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">📊</div>
            <div className="stat-content">
              <h3>Active Today</h3>
              <p className="stat-value">
                {stats.recentPredictions.filter(p => {
                  const today = new Date()
                  const predDate = new Date(p.time)
                  return predDate.toDateString() === today.toDateString()
                }).length}
              </p>
            </div>
          </div>
        </div>

        {/* Predictions by User */}
        <div className="admin-section">
          <h2>Predictions by User</h2>
          {loading ? (
            <p>Loading...</p>
          ) : Object.keys(predictionsByUser).length === 0 ? (
            <div className="empty-state">
              <p>No predictions yet.</p>
            </div>
          ) : (
            <div className="user-predictions">
              {Object.entries(predictionsByUser).map(([username, predictions]) => (
                <div key={username} className="user-prediction-group">
                  <h3 className="user-group-header">
                    <span className="user-icon">👤</span>
                    {username}
                    <span className="prediction-count">({predictions.length} predictions)</span>
                  </h3>
                  <div className="predictions-table">
                    <table>
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>Disease</th>
                          <th>Confidence</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {predictions.slice(0, 5).map((pred, idx) => (
                          <tr key={idx}>
                            <td>{formatDate(pred.time)}</td>
                            <td>{pred.label}</td>
                            <td>{Math.round(pred.probability * 100)}%</td>
                            <td>
                              <span className={`status-badge ${pred.label === 'Healthy' ? 'healthy' : 'disease'}`}>
                                {pred.label === 'Healthy' ? 'Healthy' : 'Disease'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {predictions.length > 5 && (
                      <p className="more-predictions">+ {predictions.length - 5} more predictions</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Quick Actions */}
        <div className="admin-section">
          <h2>Quick Actions</h2>
          <div className="action-buttons">
            <button onClick={() => navigate('/')} className="action-btn">
              View User Dashboard
            </button>
            <button onClick={() => navigate('/predict')} className="action-btn">
              Test Prediction
            </button>
            <button onClick={() => navigate('/history')} className="action-btn">
              View All History
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AdminDashboard


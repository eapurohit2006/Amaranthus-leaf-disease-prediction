import { Link, useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import './Home.css'

function Home() {
  const navigate = useNavigate()
  const [history, setHistory] = useState([])
  const [username, setUsername] = useState('user')

  useEffect(() => {
    // Get username from localStorage (if logged in)
    const token = localStorage.getItem('token')
    const storedUsername = localStorage.getItem('username')
    
    if (token && storedUsername) {
      setUsername(storedUsername)
      // Load history from localStorage only if logged in
      const hist = JSON.parse(localStorage.getItem('history') || '[]')
      // Filter history for current user
      const userHist = hist.filter(item => item.username === storedUsername)
      setHistory(userHist.slice(0, 5)) // Show only latest 5
    } else {
      setUsername('Guest')
      setHistory([]) // No history for guests
    }
  }, [])

  const formatDate = (timestamp) => {
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

  const getDiseaseBadge = (label) => {
    if (label === 'Healthy') {
      return <span className="disease-badge healthy">Healthy</span>
    }
    return <span className="disease-badge disease">Disease</span>
  }

  return (
    <div className="home-page">
      <div className="home-container">
        {/* Welcome Section */}
        <div className="welcome-card">
          <div className="welcome-content">
            <div className="welcome-icon">🍃</div>
            <div className="welcome-text">
              <h1>{localStorage.getItem('token') ? `Welcome back, ${username}!` : 'Welcome to LeafScan!'}</h1>
              <p>AI-powered amaranthus disease detection at your fingertips. Upload leaf images to get instant diagnoses and treatment recommendations.</p>
              {!localStorage.getItem('token') && (
                <p style={{ fontSize: '0.9rem', color: 'var(--medium-gray)', marginBottom: '1rem' }}>
                  💡 <strong>Tip:</strong> Login to save your prediction history and track plant health over time.
                </p>
              )}
              <Link to="/predict" className="analyze-btn">
                <span className="btn-icon">↑</span>
                Analyze New Leaf
              </Link>
            </div>
          </div>
        </div>

        {/* Prediction History Section */}
        <div className="history-section">
          <div className="section-header">
            <span className="section-icon">📋</span>
            <h2>Prediction History</h2>
          </div>
          <p className="section-subtitle">View your past leaf analyses and their results</p>
          
          {!localStorage.getItem('token') ? (
            <div className="empty-history">
              <div className="empty-icon">🔒</div>
              <h3>Login to view your history</h3>
              <p>Create an account or login to save and track your prediction history. You can still analyze leaves without logging in, but your history won't be saved.</p>
              <Link to="/login" className="get-started-btn">
                <span className="btn-icon">🔑</span>
                Login Now
              </Link>
            </div>
          ) : history.length === 0 ? (
            <div className="empty-history">
              <div className="empty-icon">🍃</div>
              <h3>No predictions yet.</h3>
              <p>Start analyzing amaranthus leaves to build your prediction history and track plant health over time.</p>
              <Link to="/predict" className="get-started-btn">
                <span className="btn-icon">↑</span>
                Get Started
              </Link>
            </div>
          ) : (
            <div className="history-list">
              {history.map((item, idx) => (
                <div key={idx} className="history-card">
                  <div className="history-card-header">
                    <div className="disease-info">
                      <span className="disease-icon">⚠️</span>
                      <span className="disease-name">{item.label}</span>
                    </div>
                    {getDiseaseBadge(item.label)}
                  </div>
                  <div className="history-card-meta">
                    <span className="meta-item">
                      <span className="meta-icon">📅</span>
                      {formatDate(item.time)}
                    </span>
                    <span className="meta-item">
                      <span className="meta-icon">📊</span>
                      Confidence: {Math.round(item.probability * 100)}%
                    </span>
                  </div>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill" 
                      style={{ width: `${item.probability * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Home

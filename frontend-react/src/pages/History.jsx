import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import './History.css'

function History() {
  const [items, setItems] = useState([])
  const [isLoggedIn, setIsLoggedIn] = useState(false)

  useEffect(() => {
    const token = localStorage.getItem('token')
    const username = localStorage.getItem('username')
    setIsLoggedIn(!!token)
    
    if (token && username) {
      // Only load history for logged-in users
      const data = JSON.parse(localStorage.getItem('history') || '[]')
      // Filter history for current user
      const userHistory = data.filter(item => item.username === username)
      setItems(userHistory)
    } else {
      setItems([])
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
    <div className="history-page">
      <div className="history-container">
        <div className="history-header">
          <div className="section-header">
            <span className="section-icon">📋</span>
            <h1>Prediction History</h1>
          </div>
          <p className="section-subtitle">View your past leaf analyses and their results</p>
        </div>

        {!isLoggedIn ? (
          <div className="empty-history">
            <div className="empty-icon">🔒</div>
            <h3>Please login to view your history</h3>
            <p>Your prediction history is only saved when you're logged in. Login to see your past analyses.</p>
            <Link to="/login" className="get-started-btn">
              <span className="btn-icon">🔑</span>
              Login Now
            </Link>
          </div>
        ) : items.length === 0 ? (
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
            {items.map((item, idx) => (
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
  )
}

export default History

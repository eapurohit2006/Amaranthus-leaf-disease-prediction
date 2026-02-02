import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import axios from 'axios'
import './Auth.css'

function Signup() {
  const [formData, setFormData] = useState({ username: '', password: '' })
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      await axios.post('http://localhost:8000/auth/signup', formData)
      // Redirect to login with success message
      navigate('/login', { state: { message: 'Account created successfully! Please sign in.' } })
    } catch (error) {
      alert(error.response?.data?.detail || 'Signup failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-header">
        <div className="auth-logo">
          <div className="auth-logo-icon">🍃</div>
          <h1>LeafScan</h1>
          <p>AI-Powered Plant Disease Detection</p>
        </div>
      </div>
      
      <div className="auth-card">
        <h2>Create an account</h2>
        <p className="auth-subtitle">Join LeafScan to start detecting plant diseases with AI</p>
        
        <form onSubmit={handleSubmit}>
          <div className="auth-input-group">
            <label>Username</label>
            <input
              type="text"
              placeholder="Choose a username"
              value={formData.username}
              onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              required
            />
          </div>
          
          <div className="auth-input-group">
            <label>Password</label>
            <input
              type="password"
              placeholder="Create a password (min 6 characters)"
              value={formData.password}
              onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              minLength={6}
              required
            />
          </div>
          
          <button type="submit" className="auth-submit-btn" disabled={loading}>
            {loading ? 'Creating...' : 'Create Account'}
          </button>
          
          <p className="auth-switch">
            Already have an account? <Link to="/login">Sign in</Link>
          </p>
        </form>
      </div>
    </div>
  )
}

export default Signup

import { Navigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import axios from 'axios'

function ProtectedRoute({ children, requireAdmin = false }) {
  const [isAuthenticated, setIsAuthenticated] = useState(null)
  const [userRole, setUserRole] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('token')
      if (!token) {
        setIsAuthenticated(false)
        setLoading(false)
        return
      }

      try {
        const res = await axios.get('http://localhost:8000/auth/me', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })
        setIsAuthenticated(true)
        setUserRole(res.data.role)
      } catch (error) {
        localStorage.removeItem('token')
        localStorage.removeItem('username')
        localStorage.removeItem('role')
        setIsAuthenticated(false)
      } finally {
        setLoading(false)
      }
    }

    checkAuth()
  }, [])

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh' 
      }}>
        <div>Loading...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  if (requireAdmin && userRole !== 'admin') {
    return <Navigate to="/" replace />
  }

  return children
}

export default ProtectedRoute







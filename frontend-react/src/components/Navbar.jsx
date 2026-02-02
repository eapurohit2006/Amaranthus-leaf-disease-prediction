import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useState, useEffect } from 'react'
import ProfileDropdown from './ProfileDropdown'
import './Navbar.css'

function Navbar() {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [userRole, setUserRole] = useState(null)
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    const token = localStorage.getItem('token')
    const role = localStorage.getItem('role')
    setIsLoggedIn(!!token)
    setUserRole(role)
  }, [location])

  // Don't show navbar on signup page (keep it on login since it's in navbar now)
  if (location.pathname === '/signup') {
    return null
  }

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <span className="logo-icon">🍃</span>
          <span>LeafScan</span>
        </Link>
        <div className="navbar-menu">
          {userRole === 'admin' && (
            <Link 
              to="/admin-dashboard" 
              className={location.pathname === '/admin-dashboard' ? 'nav-link active' : 'nav-link'}
            >
              Admin Dashboard
            </Link>
          )}
          <Link 
            to="/" 
            className={location.pathname === '/' ? 'nav-link active' : 'nav-link'}
          >
            Dashboard
          </Link>
          <Link 
            to="/predict" 
            className={location.pathname === '/predict' ? 'nav-link active' : 'nav-link'}
          >
            New Prediction
          </Link>
          <Link 
            to="/about" 
            className={location.pathname === '/about' ? 'nav-link active' : 'nav-link'}
          >
            About
          </Link>
          <Link 
            to="/help" 
            className={location.pathname === '/help' ? 'nav-link active' : 'nav-link'}
          >
            Help
          </Link>
          {!isLoggedIn && (
            <Link 
              to="/login" 
              className={location.pathname === '/login' ? 'nav-link active' : 'nav-link'}
            >
              Login
            </Link>
          )}
          {isLoggedIn ? (
            <ProfileDropdown />
          ) : null}
        </div>
      </div>
    </nav>
  )
}

export default Navbar

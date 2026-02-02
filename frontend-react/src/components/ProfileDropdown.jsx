import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import './ProfileDropdown.css'

function ProfileDropdown() {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef(null)
  const navigate = useNavigate()
  const username = localStorage.getItem('username') || 'User'

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('username')
    localStorage.removeItem('role')
    navigate('/login')
  }

  return (
    <div className="profile-dropdown-container" ref={dropdownRef}>
      <button 
        className="profile-avatar-btn"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Profile menu"
      >
        {username.charAt(0).toUpperCase()}
      </button>
      
      {isOpen && (
        <div className="profile-dropdown">
          <div className="profile-info">
            <div className="profile-username">{username}</div>
            <div className="profile-status">Logged in</div>
          </div>
          <div className="profile-divider"></div>
          <button className="profile-logout-btn" onClick={handleLogout}>
            <span className="logout-icon">↪</span>
            Logout
          </button>
        </div>
      )}
    </div>
  )
}

export default ProfileDropdown







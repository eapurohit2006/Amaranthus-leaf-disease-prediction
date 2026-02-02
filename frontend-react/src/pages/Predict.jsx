import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import './Predict.css'

function Predict() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [loading, setLoading] = useState(false)
  const fileInputRef = useRef(null)
  const navigate = useNavigate()

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (selectedFile) => {
    if (selectedFile.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB')
      return
    }

    if (!['image/jpeg', 'image/jpg', 'image/png'].includes(selectedFile.type)) {
      alert('Please upload a JPG, PNG, or JPEG image')
      return
    }

    setFile(selectedFile)
    const reader = new FileReader()
    reader.onloadend = () => setPreview(reader.result)
    reader.readAsDataURL(selectedFile)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) return

    const token = localStorage.getItem('token')
    const isLoggedIn = !!token

    setLoading(true)
    const formData = new FormData()
    formData.append('image', file)

    try {
      const headers = {}

      // Add auth header only if logged in
      if (isLoggedIn) {
        headers['Authorization'] = `Bearer ${token}`
      }

      // Don't set Content-Type - axios will set it automatically with boundary for FormData
      const res = await axios.post('http://localhost:8000/predict', formData, {
        headers,
        maxContentLength: Infinity,
        maxBodyLength: Infinity
      })

      // Save to history only if logged in
      // Save to history only if logged in
      if (isLoggedIn) {
        try {
          const username = localStorage.getItem('username')
          const hist = JSON.parse(localStorage.getItem('history') || '[]')
          const top = res.data.predictions[0]
          // Don't save the image to save space, just save the metadata
          hist.unshift({
            // image: preview, // Removed to prevent quota exceeded error
            label: top.label,
            probability: top.probability,
            time: Date.now(),
            precautions: top.precautions || [],
            username: username
          })
          // Limit to last 50 items
          localStorage.setItem('history', JSON.stringify(hist.slice(0, 50)))
        } catch (storageError) {
          console.error('Failed to save history:', storageError)
          // Continue execution - don't crash the prediction flow just because history save failed
        }
      }

      // Navigate to results page
      navigate('/results', {
        state: {
          predictions: res.data.predictions,
          image: preview,
          fileName: file.name,
          fileSize: file.size,
          isLoggedIn: isLoggedIn
        }
      })
    } catch (error) {
      if (error.response?.status === 401) {
        alert('Session expired. Please login again.')
        localStorage.removeItem('token')
        localStorage.removeItem('username')
        localStorage.removeItem('role')
      } else {
        const errorMsg = error.response?.data?.detail || error.message || 'Prediction failed'
        console.error('Prediction error:', error)
        console.error('Error response:', error.response?.data)
        alert(`Prediction failed: ${errorMsg}`)
      }
    } finally {
      setLoading(false)
    }
  }

  const removeImage = () => {
    setFile(null)
    setPreview(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  return (
    <div className="predict-page">
      <div className="predict-container">
        <div className="predict-header">
          <h1>Analyze Your Amaranthus Leaf</h1>
          <p>Upload a clear photo of an amaranthus leaf to detect diseases and get treatment recommendations</p>
        </div>

        {!preview ? (
          <form
            className="upload-area"
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div
              className={`upload-box ${dragActive ? 'drag-active' : ''}`}
            >
              <div className="upload-icon">↑</div>
              <h3>Upload Amaranthus Leaf Image</h3>
              <p>Drag and drop an image here, or click to browse</p>
              <p className="upload-hint">Supports: JPG, PNG, JPEG (Max 10MB)</p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/jpg,image/png"
                onChange={handleFileInput}
                style={{ display: 'none' }}
                id="file-upload"
              />
              <label htmlFor="file-upload" className="upload-browse-btn">
                Browse Files
              </label>
            </div>
          </form>
        ) : (
          <div className="upload-result">
            <div className="image-preview-card">
              <button className="remove-image-btn" onClick={removeImage}>×</button>
              <img src={preview} alt="Preview" />
              <div className="file-info">
                <span className="file-icon">📄</span>
                <span>{file.name} ({formatFileSize(file.size)})</span>
              </div>
            </div>

            <form onSubmit={handleSubmit}>
              <div className="result-actions">
                <button type="button" onClick={removeImage} className="action-btn secondary">
                  <span className="btn-icon">↻</span>
                  Remove Image
                </button>
                <button type="submit" disabled={loading} className="action-btn primary">
                  {loading ? 'Analyzing...' : (
                    <>
                      <span className="btn-icon">🔍</span>
                      Analyze Image
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        )}
      </div>
    </div>
  )
}

export default Predict

import { useLocation, useNavigate } from 'react-router-dom'
import { jsPDF } from 'jspdf'
import html2canvas from 'html2canvas'
import './Results.css'

function Results() {
  const { state } = useLocation()
  const navigate = useNavigate()
  const isLoggedIn = state?.isLoggedIn ?? !!localStorage.getItem('token')

  if (!state?.predictions) {
    return (
      <div className="results-page">
        <div className="results-container">
          <p>No results found. Please upload an image first.</p>
          <button onClick={() => navigate('/predict')} className="action-btn primary">
            Go to Predict
          </button>
        </div>
      </div>
    )
  }

  const topPrediction = state.predictions[0]
  const confidence = Math.round(topPrediction.probability * 100)
  const isHealthy = topPrediction.label === 'Healthy'

  const formatFileSize = (bytes) => {
    if (!bytes) return ''
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  const downloadReport = async () => {
    try {
      // Create a comprehensive report
      const pdf = new jsPDF('p', 'mm', 'a4')
      const pageWidth = pdf.internal.pageSize.getWidth()
      const pageHeight = pdf.internal.pageSize.getHeight()
      const margin = 15
      let yPosition = margin

      // Add header
      pdf.setFillColor(40, 167, 69) // Primary green
      pdf.rect(0, 0, pageWidth, 30, 'F')
      pdf.setTextColor(255, 255, 255)
      pdf.setFontSize(20)
      pdf.setFont('helvetica', 'bold')
      pdf.text('LeafScan Disease Detection Report', pageWidth / 2, 18, { align: 'center' })
      
      pdf.setTextColor(108, 117, 125) // Medium gray
      pdf.setFontSize(10)
      pdf.setFont('helvetica', 'normal')
      pdf.text(`Generated on ${new Date().toLocaleString()}`, pageWidth / 2, 25, { align: 'center' })
      
      yPosition = 40
      pdf.setTextColor(33, 37, 41) // Dark gray

      // Add disease label
      pdf.setFontSize(16)
      pdf.setFont('helvetica', 'bold')
      pdf.text(`Disease Detected: ${topPrediction.label}`, margin, yPosition)
      yPosition += 10

      // Add confidence level
      pdf.setFontSize(12)
      pdf.setFont('helvetica', 'normal')
      pdf.text(`Confidence Level: ${confidence}%`, margin, yPosition)
      yPosition += 8

      // Draw confidence bar
      const barWidth = pageWidth - (margin * 2)
      const barHeight = 5
      pdf.setFillColor(248, 249, 250) // Light gray
      pdf.rect(margin, yPosition, barWidth, barHeight, 'F')
      pdf.setFillColor(40, 167, 69) // Green
      pdf.rect(margin, yPosition, (barWidth * confidence) / 100, barHeight, 'F')
      yPosition += 12

      // Add treatment recommendations
      if (topPrediction.precautions && topPrediction.precautions.length > 0) {
        pdf.setFontSize(14)
        pdf.setFont('helvetica', 'bold')
        pdf.text('Treatment Recommendations:', margin, yPosition)
        yPosition += 8

        pdf.setFontSize(10)
        pdf.setFont('helvetica', 'normal')
        topPrediction.precautions.forEach((prec, idx) => {
          if (yPosition > pageHeight - 20) {
            pdf.addPage()
            yPosition = margin
          }
          pdf.text(`${idx + 1}. ${prec}`, margin + 5, yPosition)
          yPosition += 6
        })
        yPosition += 5
      }

      // Add note
      if (yPosition > pageHeight - 30) {
        pdf.addPage()
        yPosition = margin
      }
      pdf.setFontSize(9)
      pdf.setTextColor(108, 117, 125)
      pdf.setFont('helvetica', 'italic')
      const noteText = 'Note: This is an AI-powered prediction. For critical cases or confirmation, please consult with a professional agronomist or plant pathologist.'
      const splitNote = pdf.splitTextToSize(noteText, pageWidth - (margin * 2))
      pdf.text(splitNote, margin, yPosition)
      yPosition += splitNote.length * 5 + 5

      // Add image info
      if (state.fileName) {
        if (yPosition > pageHeight - 20) {
          pdf.addPage()
          yPosition = margin
        }
        pdf.setFontSize(10)
        pdf.setFont('helvetica', 'normal')
        pdf.setTextColor(108, 117, 125)
        pdf.text(`Image: ${state.fileName}`, margin, yPosition)
        if (state.fileSize) {
          yPosition += 5
          pdf.text(`File Size: ${formatFileSize(state.fileSize)}`, margin, yPosition)
        }
      }

      // Save PDF
      const fileName = `LeafScan_Report_${topPrediction.label.replace(/\s+/g, '_')}_${Date.now()}.pdf`
      pdf.save(fileName)
    } catch (error) {
      console.error('Error generating report:', error)
      alert('Failed to generate report. Please try again.')
    }
  }

  return (
    <div className="results-page">
      <div className="results-container">
        {/* Image Display */}
        <div className="image-display-card">
          <button className="remove-image-btn" onClick={() => navigate('/predict')}>×</button>
          <img src={state.image} alt="Amaranthus leaf" />
          {state.fileName && (
            <div className="file-info">
              <span className="file-icon">📄</span>
              <span>{state.fileName} {state.fileSize && `(${formatFileSize(state.fileSize)})`}</span>
            </div>
          )}
        </div>

        {/* Results Card */}
        <div className="result-card">
          <div className="result-header">
            <span className="result-icon">{isHealthy ? '✅' : '⚠️'}</span>
            <h2>{topPrediction.label}</h2>
            {!isHealthy && <span className="disease-badge">Disease Detected</span>}
          </div>

          <div className="confidence-section">
            <p className="confidence-label">Confidence Level</p>
            <div className="confidence-display">
              <div className="confidence-bar">
                <div className="confidence-fill" style={{ width: `${confidence}%` }}></div>
              </div>
              <span className="confidence-percentage">{confidence}%</span>
            </div>
            <p className="confidence-note">
              {confidence >= 80 ? 'High confidence in prediction' : 
               confidence >= 60 ? 'Moderate confidence in prediction' : 
               'Low confidence in prediction'}
            </p>
          </div>

          {topPrediction.precautions && topPrediction.precautions.length > 0 && (
            <div className="treatment-section">
              <h3>
                <span className="treatment-icon">⚠️</span>
                Treatment Recommendations
              </h3>
              <ul className="treatment-list">
                {topPrediction.precautions.map((prec, idx) => (
                  <li key={idx}>{prec}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="result-note">
            Note: This is an AI-powered prediction. For critical cases or confirmation, please consult with a professional agronomist or plant pathologist.
          </div>

          <div className="result-actions">
            <button onClick={() => navigate('/predict')} className="action-btn secondary">
              <span className="btn-icon">↻</span>
              Analyze Another
            </button>
            <button onClick={downloadReport} className="action-btn primary">
              <span className="btn-icon">📥</span>
              Download Report
            </button>
            {isLoggedIn && (
              <button onClick={() => navigate('/history')} className="action-btn primary">
                <span className="btn-icon">📋</span>
                View History
              </button>
            )}
            {!isLoggedIn && (
              <button onClick={() => navigate('/login')} className="action-btn secondary">
                <span className="btn-icon">🔑</span>
                Login to Save History
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Results

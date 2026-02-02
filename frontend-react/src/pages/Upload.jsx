import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import './Upload.css'
import { toast } from '../components/Toast'

function Upload() {
	const [file, setFile] = useState(null)
	const [preview, setPreview] = useState(null)
	const [usingCam, setUsingCam] = useState(false)
	const videoRef = useRef(null)
	const canvasRef = useRef(null)
	const navigate = useNavigate()

	useEffect(() => {
		if (!usingCam) return
		navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
			videoRef.current.srcObject = stream
			videoRef.current.play()
		}).catch(() => {
			toast('Camera permission denied', 'error')
			setUsingCam(false)
		})
		return () => {
			if (videoRef.current && videoRef.current.srcObject) {
				videoRef.current.srcObject.getTracks().forEach((t) => t.stop())
			}
		}
	}, [usingCam])

	const onDrop = (e) => {
		e.preventDefault()
		const f = e.dataTransfer.files?.[0]
		if (f) handleFile(f)
	}

	const handleFile = (f) => {
		if (!f.type.startsWith('image/')) {
			toast('Please upload an image', 'error')
			return
		}
		setFile(f)
		const reader = new FileReader()
		reader.onloadend = () => setPreview(reader.result)
		reader.readAsDataURL(f)
	}

	const capture = () => {
		const video = videoRef.current
		const canvas = canvasRef.current
		canvas.width = video.videoWidth
		canvas.height = video.videoHeight
		const ctx = canvas.getContext('2d')
		ctx.drawImage(video, 0, 0)
		canvas.toBlob((blob) => {
			if (blob) {
				const f = new File([blob], 'capture.jpg', { type: 'image/jpeg' })
				handleFile(f)
			}
		})
	}

	const goPredict = () => {
		if (!file) return toast('Please select an image first', 'error')
		navigate('/predict', { state: { file, preview } })
	}

	return (
		<div className="upload-page">
			<section className="upload-hero">
				<div className="container">
					<h1>Scan Leaf Now</h1>
					<p>Upload or capture a clear image of the leaf for best results</p>
				</div>
			</section>

			<section className="upload-content container">
				<div className="uploader">
					<div className="dropzone" onDragOver={(e)=>e.preventDefault()} onDrop={onDrop}>
						<input type="file" accept="image/*" id="file-input" onChange={(e)=>e.target.files[0] && handleFile(e.target.files[0])} />
						<label htmlFor="file-input" className="drop-label">
							<span className="drop-icon">⬆️</span>
							<span className="drop-text">Drag & drop or click to upload</span>
						</label>
					</div>

					<div className="or">OR</div>

					<div className="camera">
						{usingCam ? (
							<>
								<video ref={videoRef} className="video" />
								<canvas ref={canvasRef} className="hidden" />
								<button className="btn btn-primary" onClick={capture}>Capture</button>
								<button className="btn" onClick={()=>setUsingCam(false)}>Close Camera</button>
							</>
						) : (
							<button className="btn btn-primary" onClick={()=>setUsingCam(true)}>Use Camera</button>
						)}
					</div>
				</div>

				{preview && (
					<div className="preview-card">
						<h3>Preview</h3>
						<img src={preview} alt="preview" />
					</div>
				)}

				<div className="actions">
					<button className="btn btn-primary btn-large" onClick={goPredict}>Predict</button>
					<a className="btn" href="/help">How to take a good photo</a>
				</div>
			</section>
		</div>
	)
}

export default Upload












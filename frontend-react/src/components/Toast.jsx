import { useEffect, useState } from 'react'

let pushToast = () => {}

export function ToastHost() {
	const [toasts, setToasts] = useState([])
	useEffect(() => {
		pushToast = (t) => {
			const id = Math.random().toString(36).slice(2)
			setToasts((x) => [...x, { id, ...t }])
			setTimeout(() => setToasts((x) => x.filter((y) => y.id !== id)), t.duration || 3000)
		}
	}, [])
	return (
		<div style={{ position: 'fixed', right: 16, bottom: 16, display: 'grid', gap: 8, zIndex: 2000 }}>
			{toasts.map((t) => (
				<div key={t.id} style={{ background: t.type === 'error' ? '#ffebee' : '#e8f5e9', color: '#2c3e50', padding: '12px 16px', borderRadius: 12, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
					{t.message}
				</div>
			))}
		</div>
	)
}

export function toast(message, type = 'info', duration = 3000) {
	pushToast({ message, type, duration })
}












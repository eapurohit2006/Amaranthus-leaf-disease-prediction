import { createContext, useContext, useEffect, useState } from 'react'

const ThemeCtx = createContext({ theme: 'light', toggle: () => {} })

export function ThemeProvider({ children }) {
	const [theme, setTheme] = useState(localStorage.getItem('theme') || 'light')
	useEffect(() => {
		localStorage.setItem('theme', theme)
		document.documentElement.setAttribute('data-theme', theme)
	}, [theme])
	const toggle = () => setTheme(theme === 'light' ? 'dark' : 'light')
	return <ThemeCtx.Provider value={{ theme, toggle }}>{children}</ThemeCtx.Provider>
}

export function useTheme() {
	return useContext(ThemeCtx)
}












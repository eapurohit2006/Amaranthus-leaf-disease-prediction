import express from 'express'
import cors from 'cors'
import axios from 'axios'

const app = express()
const PORT = 8001
const PYTHON_API = 'http://localhost:8000'

app.use(cors())
app.use(express.json())

// Proxy to Python FastAPI backend
app.use('/auth', async (req, res) => {
  try {
    const response = await axios({
      method: req.method,
      url: `${PYTHON_API}${req.path}`,
      data: req.body,
      headers: req.headers
    })
    res.json(response.data)
  } catch (error) {
    res.status(error.response?.status || 500).json(error.response?.data || { error: 'Internal error' })
  }
})

app.use('/predict', async (req, res) => {
  try {
    const response = await axios({
      method: req.method,
      url: `${PYTHON_API}${req.path}`,
      data: req.body,
      headers: req.headers,
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    })
    res.json(response.data)
  } catch (error) {
    res.status(error.response?.status || 500).json(error.response?.data || { error: 'Internal error' })
  }
})

app.listen(PORT, () => {
  console.log(`Node.js proxy server running on http://localhost:${PORT}`)
})












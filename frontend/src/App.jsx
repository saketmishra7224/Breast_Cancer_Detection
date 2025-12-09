import React from 'react'
import { Routes, Route } from 'react-router-dom'
import SinglePredict from './pages/SinglePredict'

export default function App() {
  return (
    <div style={{ fontFamily: 'Inter, system-ui, sans-serif', padding: 20 }}>
      <header style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <h2>Breast Cancer Detection</h2>
      </header>
      <main style={{ marginTop: 20 }}>
        <Routes>
          <Route path="/" element={<SinglePredict />} />
        </Routes>
      </main>
    </div>
  )
}

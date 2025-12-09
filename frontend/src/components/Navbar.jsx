import React from 'react'
import { Link } from 'react-router-dom'

export function Navbar() {
  return (
    <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div style={{ fontSize: 18, fontWeight: 600 }}>Breast Cancer Detection</div>
      <nav style={{ display: 'flex', gap: 12 }}>
        <Link to="/">Single</Link>
      </nav>
    </header>
  )
}

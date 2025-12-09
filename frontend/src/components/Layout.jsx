import React from 'react'

export function Layout({ children }) {
  return (
    <div style={{ minHeight: '100vh', background: '#0f1724', color: '#e6eef8', padding: 20 }}>
      <div style={{ maxWidth: 900, margin: '0 auto' }}>{children}</div>
    </div>
  )
}

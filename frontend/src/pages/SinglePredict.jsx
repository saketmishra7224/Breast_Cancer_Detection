import React from 'react'
import { Layout } from '../components/Layout'
import { Navbar } from '../components/Navbar'
import FeatureForm from '../components/FeatureForm'

export default function SinglePredict() {
  return (
    <Layout>
      <Navbar />
      <main style={{ marginTop: 16 }}>
        <h1>Single Patient Prediction</h1>
        <p style={{ color: '#9ca3af' }}>Enter diagnostic features to estimate probability. Educational use only.</p>
        <FeatureForm />
      </main>
    </Layout>
  )
}

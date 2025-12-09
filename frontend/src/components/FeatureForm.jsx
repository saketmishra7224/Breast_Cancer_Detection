import React, { useEffect, useState } from 'react'
import api from '../api/client'

export default function FeatureForm() {
  const [features, setFeatures] = useState([])
  const [values, setValues] = useState({})
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  useEffect(() => {
    api.get('/features').then(res => {
      setFeatures(res.data)
      const init = {}
      res.data.forEach(f => (init[f] = ''))
      setValues(init)
    }).catch(() => {})
  }, [])

  const onChange = (name, value) => setValues(prev => ({ ...prev, [name]: value }))

  const onSubmit = async e => {
    e.preventDefault()
    setLoading(true)
    try {
      const numericFeatures = {}
      for (const k of Object.keys(values)) numericFeatures[k] = parseFloat(values[k])
      const res = await api.post('/predict', { features: numericFeatures })
      setResult(res.data)
    } catch (err) {
      console.error(err)
      alert('Prediction failed: ' + (err?.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  if (!features.length) return <div>Loading features...</div>

  return (
    <div>
      <form onSubmit={onSubmit} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {features.map(name => (
          <div key={name} style={{ display: 'flex', flexDirection: 'column' }}>
            <label style={{ fontSize: 12, color: '#9ca3af' }}>{name}</label>
            <input value={values[name] ?? ''} onChange={e => onChange(name, e.target.value)} type="number" step="any" />
          </div>
        ))}
        <div style={{ gridColumn: '1 / -1', marginTop: 12 }}>
          <button type="submit" disabled={loading}>{loading ? 'Predicting...' : 'Predict'}</button>
        </div>
      </form>

      {result && (
        <div style={{ marginTop: 12, padding: 12, background: '#0b1220', borderRadius: 8 }}>
          <div>Predicted class: <strong>{result.prediction === 1 ? 'Benign (1)' : 'Malignant (0)'}</strong></div>
          <div>Probability benign: {(result.probability_benign * 100).toFixed(2)}%</div>
          <div>Probability malignant: {(result.probability_malignant * 100).toFixed(2)}%</div>
        </div>
      )}
    </div>
  )
}

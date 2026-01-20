import React, { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, Brain, AlertCircle, Check, User } from 'lucide-react'
import api from '../services/api'

function Prediction() {
    const navigate = useNavigate()
    const fileInputRef = useRef(null)
    const [step, setStep] = useState(1)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    // Simplified patient info - only ID, name, age
    const [patient, setPatient] = useState({
        patient_id: '',
        name: '',
        age: ''
    })

    const [selectedFile, setSelectedFile] = useState(null)
    const [preview, setPreview] = useState(null)
    const [result, setResult] = useState(null)

    const handlePatientChange = (e) => {
        setPatient({ ...patient, [e.target.name]: e.target.value })
    }

    const handleFileSelect = (e) => {
        const file = e.target.files[0]
        if (file) {
            setSelectedFile(file)
            const reader = new FileReader()
            reader.onloadend = () => setPreview(reader.result)
            reader.readAsDataURL(file)
        }
    }

    const handleDrop = (e) => {
        e.preventDefault()
        const file = e.dataTransfer.files[0]
        if (file && file.type.startsWith('image/')) {
            setSelectedFile(file)
            const reader = new FileReader()
            reader.onloadend = () => setPreview(reader.result)
            reader.readAsDataURL(file)
        }
    }

    const handleSubmit = async () => {
        if (!selectedFile) {
            setError('Please upload an MRI image')
            return
        }

        if (!patient.patient_id || !patient.name || !patient.age) {
            setError('Please fill in all patient information')
            return
        }

        setLoading(true)
        setError('')

        try {
            // Step 1: Create or get patient
            let patientDbId
            try {
                const patientRes = await api.post('/api/patients/', {
                    patient_id: patient.patient_id,
                    name: patient.name,
                    age: parseInt(patient.age)
                })
                patientDbId = patientRes.data.id
            } catch (err) {
                if (err.response?.status === 400 && err.response?.data?.detail?.includes('exists')) {
                    // Patient exists, search for them
                    const searchRes = await api.get(`/api/patients/search/${patient.patient_id}`)
                    if (searchRes.data.length > 0) {
                        patientDbId = searchRes.data[0].id
                    } else {
                        throw new Error('Patient ID exists but could not be found')
                    }
                } else {
                    throw err
                }
            }

            // Step 2: Create prediction with real API call
            const formData = new FormData()
            formData.append('patient_id', patientDbId)
            formData.append('image', selectedFile)

            const predictionRes = await api.post('/api/predictions/', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            })

            setResult(predictionRes.data)
            setStep(3)

        } catch (err) {
            console.error('Prediction error:', err)
            setError(err.response?.data?.detail || 'Prediction failed. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    const getResultColor = (cls) => {
        const colors = {
            'NonDemented': '#10B981',
            'VeryMildDemented': '#F59E0B',
            'MildDemented': '#F97316',
            'ModerateDemented': '#EF4444'
        }
        return colors[cls] || '#6B7280'
    }

    const getResultDescription = (cls) => {
        const descriptions = {
            'NonDemented': 'Cognitively Normal - No significant signs of dementia detected',
            'VeryMildDemented': 'Very Mild Cognitive Impairment - Early changes warrant monitoring',
            'MildDemented': 'Mild Dementia - Consistent with early-stage Alzheimer\'s',
            'ModerateDemented': 'Moderate Dementia - Significant cognitive impairment'
        }
        return descriptions[cls] || ''
    }

    return (
        <div style={{ maxWidth: '900px', margin: '0 auto' }}>
            {/* Progress Steps */}
            <div style={{
                display: 'flex',
                justifyContent: 'center',
                marginBottom: '2rem',
                gap: '1rem'
            }}>
                {[1, 2, 3].map(s => (
                    <div key={s} style={{ display: 'flex', alignItems: 'center' }}>
                        <div style={{
                            width: '32px',
                            height: '32px',
                            borderRadius: '50%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: step >= s ? 'var(--gradient-primary)' : 'var(--light)',
                            color: step >= s ? 'white' : 'var(--gray)',
                            fontWeight: '600',
                            fontSize: '0.875rem'
                        }}>
                            {step > s ? <Check size={18} /> : s}
                        </div>
                        <span style={{
                            marginLeft: '0.5rem',
                            fontSize: '0.875rem',
                            color: step >= s ? 'var(--dark)' : 'var(--gray)'
                        }}>
                            {s === 1 ? 'Patient' : s === 2 ? 'MRI Scan' : 'Results'}
                        </span>
                        {s < 3 && (
                            <div style={{
                                width: '60px',
                                height: '2px',
                                background: step > s ? 'var(--primary)' : 'var(--light)',
                                margin: '0 1rem'
                            }} />
                        )}
                    </div>
                ))}
            </div>

            {error && (
                <div style={{
                    background: 'rgba(239, 68, 68, 0.1)',
                    color: 'var(--danger)',
                    padding: '1rem',
                    borderRadius: '0.5rem',
                    marginBottom: '1.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem'
                }}>
                    <AlertCircle size={20} />
                    {error}
                </div>
            )}

            {/* Step 1: Patient Information (Simplified) */}
            {step === 1 && (
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            <User size={20} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
                            Patient Information
                        </h3>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                        <div className="form-group">
                            <label className="form-label">Patient ID *</label>
                            <input
                                type="text"
                                name="patient_id"
                                className="form-input"
                                placeholder="P-2024-001"
                                value={patient.patient_id}
                                onChange={handlePatientChange}
                                required
                            />
                        </div>

                        <div className="form-group">
                            <label className="form-label">Full Name *</label>
                            <input
                                type="text"
                                name="name"
                                className="form-input"
                                placeholder="John Smith"
                                value={patient.name}
                                onChange={handlePatientChange}
                                required
                            />
                        </div>

                        <div className="form-group">
                            <label className="form-label">Age *</label>
                            <input
                                type="number"
                                name="age"
                                className="form-input"
                                placeholder="65"
                                min="1"
                                max="120"
                                value={patient.age}
                                onChange={handlePatientChange}
                                required
                            />
                        </div>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1.5rem' }}>
                        <button
                            className="btn btn-primary btn-lg"
                            onClick={() => setStep(2)}
                            disabled={!patient.patient_id || !patient.name || !patient.age}
                        >
                            Continue to Upload
                        </button>
                    </div>
                </div>
            )}

            {/* Step 2: Upload MRI */}
            {step === 2 && (
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Upload MRI Scan</h3>
                        <span style={{ fontSize: '0.875rem', color: 'var(--gray)' }}>
                            Patient: {patient.name} ({patient.patient_id})
                        </span>
                    </div>

                    <div
                        className={`file-upload ${selectedFile ? 'active' : ''}`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                        onClick={() => fileInputRef.current.click()}
                    >
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="image/*"
                            onChange={handleFileSelect}
                            style={{ display: 'none' }}
                        />

                        {preview ? (
                            <div>
                                <img
                                    src={preview}
                                    alt="MRI Preview"
                                    style={{
                                        maxWidth: '300px',
                                        maxHeight: '300px',
                                        borderRadius: '0.5rem',
                                        marginBottom: '1rem'
                                    }}
                                />
                                <p style={{ color: 'var(--success)', margin: 0 }}>
                                    <Check size={18} style={{ verticalAlign: 'middle', marginRight: '0.5rem' }} />
                                    {selectedFile.name}
                                </p>
                            </div>
                        ) : (
                            <div>
                                <Upload size={48} color="var(--gray-light)" style={{ marginBottom: '1rem' }} />
                                <p style={{ fontSize: '1.125rem', color: 'var(--dark)', marginBottom: '0.5rem' }}>
                                    Drop MRI image here or click to browse
                                </p>
                                <p style={{ fontSize: '0.875rem', color: 'var(--gray)' }}>
                                    Supports: JPG, PNG
                                </p>
                            </div>
                        )}
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1.5rem' }}>
                        <button className="btn btn-secondary" onClick={() => setStep(1)}>
                            Back
                        </button>
                        <button
                            className="btn btn-primary btn-lg"
                            onClick={handleSubmit}
                            disabled={loading || !selectedFile}
                        >
                            {loading ? (
                                <>
                                    <div className="spinner" style={{ width: '20px', height: '20px', borderWidth: '2px' }} />
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <Brain size={20} />
                                    Run AI Analysis
                                </>
                            )}
                        </button>
                    </div>
                </div>
            )}

            {/* Step 3: Results */}
            {step === 3 && result && (
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Analysis Results</h3>
                        <span style={{ fontSize: '0.875rem', color: 'var(--gray)' }}>
                            {patient.name} | Age: {patient.age}
                        </span>
                    </div>

                    <div className="result-display">
                        <div style={{
                            width: '100px',
                            height: '100px',
                            borderRadius: '50%',
                            background: `${getResultColor(result.predicted_class)}20`,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            margin: '0 auto 1.5rem'
                        }}>
                            <Brain size={48} color={getResultColor(result.predicted_class)} />
                        </div>

                        <h2 style={{ color: getResultColor(result.predicted_class), marginBottom: '0.5rem' }}>
                            {result.predicted_class}
                        </h2>

                        <p style={{ color: 'var(--gray)', marginBottom: '1rem' }}>
                            {getResultDescription(result.predicted_class)}
                        </p>

                        <p style={{ fontSize: '1.25rem', marginBottom: '1.5rem' }}>
                            Confidence: <strong>{(result.confidence_score * 100).toFixed(1)}%</strong>
                        </p>
                    </div>

                    {/* Probability Breakdown */}
                    {result.probabilities && (
                        <div style={{ marginTop: '2rem' }}>
                            <h4 style={{ marginBottom: '1rem' }}>Classification Probabilities</h4>
                            {Object.entries(result.probabilities)
                                .sort((a, b) => b[1] - a[1])
                                .map(([cls, prob]) => (
                                    <div key={cls} style={{ marginBottom: '0.75rem' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                            <span style={{ fontSize: '0.875rem' }}>{cls}</span>
                                            <span style={{ fontSize: '0.875rem', fontWeight: '500' }}>{(prob * 100).toFixed(1)}%</span>
                                        </div>
                                        <div style={{ height: '8px', background: 'var(--light)', borderRadius: '4px' }}>
                                            <div style={{
                                                width: `${prob * 100}%`,
                                                height: '100%',
                                                background: getResultColor(cls),
                                                borderRadius: '4px',
                                                transition: 'width 0.5s ease'
                                            }} />
                                        </div>
                                    </div>
                                ))
                            }
                        </div>
                    )}

                    {/* Highlighted Brain Regions */}
                    {result.highlighted_regions && result.highlighted_regions.length > 0 && (
                        <div style={{ marginTop: '2rem', padding: '1rem', background: 'var(--lighter)', borderRadius: '0.5rem' }}>
                            <h4 style={{ marginBottom: '0.75rem' }}>Key Brain Regions Identified</h4>
                            <ul style={{ margin: 0, paddingLeft: '1.5rem' }}>
                                {result.highlighted_regions.map((region, idx) => (
                                    <li key={idx} style={{ marginBottom: '0.25rem', color: 'var(--gray)' }}>{region}</li>
                                ))}
                            </ul>
                        </div>
                    )}

                    <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginTop: '2rem' }}>
                        <button
                            className="btn btn-secondary"
                            onClick={() => navigate(`/results/${result.id}`)}
                        >
                            View Full Details
                        </button>
                        <button
                            className="btn btn-secondary"
                            onClick={() => navigate('/chat')}
                        >
                            Ask AI Assistant
                        </button>
                        <button
                            className="btn btn-primary"
                            onClick={() => {
                                setStep(1)
                                setPatient({ patient_id: '', name: '', age: '' })
                                setSelectedFile(null)
                                setPreview(null)
                                setResult(null)
                            }}
                        >
                            New Prediction
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}

export default Prediction

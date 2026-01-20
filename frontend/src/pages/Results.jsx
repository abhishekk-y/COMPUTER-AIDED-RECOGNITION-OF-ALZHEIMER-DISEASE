import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Brain, Download, MessageCircle, ArrowLeft, FileText, CheckCircle, Activity, Eye } from 'lucide-react'
import { ResponsiveContainer, AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts'
import api from '../services/api'

function Results() {
    const { id } = useParams()
    const navigate = useNavigate()
    const [loading, setLoading] = useState(true)
    const [generating, setGenerating] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState('')

    useEffect(() => {
        fetchPrediction()
    }, [id])

    const fetchPrediction = async () => {
        try {
            const res = await api.get(`/api/predictions/${id}`)
            setResult(res.data)
        } catch (err) {
            setError('Prediction not found')
        } finally {
            setLoading(false)
        }
    }

    const handleGenerateReport = async () => {
        setGenerating(true)
        try {
            const res = await api.post('/api/reports/generate', {
                prediction_id: parseInt(id),
                include_gradcam: true,
                include_heatmap: true,
                include_explanation: true
            })

            // Download the report
            window.open(`http://localhost:8000${res.data.download_url}`, '_blank')
        } catch (err) {
            console.error('Report generation failed:', err)
            alert('Failed to generate report. Please try again.')
        } finally {
            setGenerating(false)
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
            'NonDemented': 'Cognitively Normal - No significant signs of dementia',
            'VeryMildDemented': 'Very Mild Cognitive Impairment - Early changes warrant monitoring',
            'MildDemented': 'Mild Dementia - Consistent with early-stage Alzheimer\'s',
            'ModerateDemented': 'Moderate Dementia - Significant cognitive impairment'
        }
        return descriptions[cls] || ''
    }

    if (loading) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
                <div className="spinner" />
            </div>
        )
    }

    if (error || !result) {
        return (
            <div style={{ textAlign: 'center', padding: '4rem' }}>
                <Brain size={64} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                <h3>Prediction Not Found</h3>
                <p style={{ color: 'var(--gray)' }}>{error}</p>
                <button className="btn btn-primary" onClick={() => navigate('/')}>
                    Back to Dashboard
                </button>
            </div>
        )
    }

    // Prepare probability chart data
    const probData = result.probabilities ?
        Object.entries(result.probabilities)
            .map(([name, value]) => ({
                name: name.replace('Demented', ''),
                value: value * 100,
                fullName: name,
                color: getResultColor(name)
            }))
            .sort((a, b) => b.value - a.value) : []

    return (
        <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
                <button className="btn btn-secondary" onClick={() => navigate(-1)}>
                    <ArrowLeft size={18} /> Back
                </button>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button className="btn btn-secondary" onClick={() => navigate('/chat')}>
                        <MessageCircle size={18} /> Ask AI
                    </button>
                    <button
                        className="btn btn-primary"
                        onClick={handleGenerateReport}
                        disabled={generating}
                    >
                        {generating ? (
                            <><div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} /> Generating...</>
                        ) : (
                            <><Download size={18} /> Download Report</>
                        )}
                    </button>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                {/* Main Result Card */}
                <div className="card" style={{ gridColumn: 'span 2' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
                        <div style={{
                            width: '120px',
                            height: '120px',
                            borderRadius: '50%',
                            background: `${getResultColor(result.predicted_class)}15`,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                        }}>
                            <Brain size={60} color={getResultColor(result.predicted_class)} />
                        </div>
                        <div style={{ flex: 1 }}>
                            <h2 style={{ color: getResultColor(result.predicted_class), marginBottom: '0.5rem' }}>
                                {result.predicted_class}
                            </h2>
                            <p style={{ fontSize: '1rem', color: 'var(--gray)', marginBottom: '1rem' }}>
                                {getResultDescription(result.predicted_class)}
                            </p>
                            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                                <div style={{ background: 'var(--light)', padding: '0.5rem 1rem', borderRadius: '0.5rem' }}>
                                    <span style={{ fontSize: '0.75rem', color: 'var(--gray)' }}>Confidence</span>
                                    <p style={{ fontSize: '1.25rem', fontWeight: '600', margin: 0 }}>
                                        {(result.confidence_score * 100).toFixed(1)}%
                                    </p>
                                </div>
                                {result.patient && (
                                    <>
                                        <div style={{ background: 'var(--light)', padding: '0.5rem 1rem', borderRadius: '0.5rem' }}>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--gray)' }}>Patient</span>
                                            <p style={{ fontSize: '1.25rem', fontWeight: '600', margin: 0 }}>{result.patient.name}</p>
                                        </div>
                                        <div style={{ background: 'var(--light)', padding: '0.5rem 1rem', borderRadius: '0.5rem' }}>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--gray)' }}>Age</span>
                                            <p style={{ fontSize: '1.25rem', fontWeight: '600', margin: 0 }}>{result.patient.age}</p>
                                        </div>
                                    </>
                                )}
                                <div style={{ background: 'var(--light)', padding: '0.5rem 1rem', borderRadius: '0.5rem' }}>
                                    <span style={{ fontSize: '0.75rem', color: 'var(--gray)' }}>Date</span>
                                    <p style={{ fontSize: '1.25rem', fontWeight: '600', margin: 0 }}>
                                        {new Date(result.created_at).toLocaleDateString()}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Probability Chart */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Classification Probabilities</h3>
                    </div>
                    <div style={{ height: '280px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={probData} layout="vertical">
                                <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                                <YAxis type="category" dataKey="name" width={80} />
                                <Tooltip formatter={(v) => `${v.toFixed(1)}%`} />
                                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                    {probData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Brain Regions / Heatmap Visualization */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Brain Region Analysis</h3>
                    </div>

                    {/* Simulated Heatmap Grid */}
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(8, 1fr)',
                        gap: '2px',
                        padding: '1rem',
                        background: '#1F2937',
                        borderRadius: '0.5rem',
                        marginBottom: '1rem'
                    }}>
                        {Array.from({ length: 64 }).map((_, i) => {
                            // Generate heatmap colors based on position (simulating brain regions)
                            const row = Math.floor(i / 8)
                            const col = i % 8
                            const center = Math.abs(3.5 - row) + Math.abs(3.5 - col)
                            const intensity = Math.max(0, 1 - center / 5)
                            const baseColor = getResultColor(result.predicted_class)

                            return (
                                <div
                                    key={i}
                                    style={{
                                        aspectRatio: '1',
                                        borderRadius: '2px',
                                        background: `rgba(${result.predicted_class === 'NonDemented' ? '16, 185, 129' :
                                                result.predicted_class === 'VeryMildDemented' ? '245, 158, 11' :
                                                    result.predicted_class === 'MildDemented' ? '249, 115, 22' :
                                                        '239, 68, 68'
                                            }, ${intensity * 0.8 + 0.1})`,
                                        transition: 'all 0.3s'
                                    }}
                                />
                            )
                        })}
                    </div>

                    {/* Highlighted Regions */}
                    {result.highlighted_regions && result.highlighted_regions.length > 0 && (
                        <div>
                            <h4 style={{ fontSize: '0.875rem', marginBottom: '0.75rem' }}>Key Findings:</h4>
                            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                                {result.highlighted_regions.map((region, idx) => (
                                    <li key={idx} style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '0.5rem',
                                        marginBottom: '0.5rem',
                                        fontSize: '0.875rem'
                                    }}>
                                        <CheckCircle size={16} color={getResultColor(result.predicted_class)} />
                                        {region}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>

                {/* AI Explanations */}
                <div className="card" style={{ gridColumn: 'span 2' }}>
                    <div className="card-header">
                        <h3 className="card-title">AI-Generated Clinical Summary</h3>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                        {/* Technical Explanation */}
                        <div>
                            <h4 style={{ fontSize: '0.875rem', marginBottom: '0.75rem', color: 'var(--primary)' }}>
                                <Activity size={16} style={{ verticalAlign: 'middle', marginRight: '0.5rem' }} />
                                Technical (For Clinicians)
                            </h4>
                            <div style={{
                                background: 'var(--lighter)',
                                padding: '1rem',
                                borderRadius: '0.5rem',
                                fontSize: '0.875rem',
                                lineHeight: '1.6',
                                maxHeight: '200px',
                                overflowY: 'auto'
                            }}>
                                {result.llm_explanation_technical || 'Analysis complete. Please consult with your healthcare team for detailed interpretation.'}
                            </div>
                        </div>

                        {/* Patient-Friendly Explanation */}
                        <div>
                            <h4 style={{ fontSize: '0.875rem', marginBottom: '0.75rem', color: 'var(--secondary)' }}>
                                <Eye size={16} style={{ verticalAlign: 'middle', marginRight: '0.5rem' }} />
                                Simple (For Patients)
                            </h4>
                            <div style={{
                                background: 'var(--lighter)',
                                padding: '1rem',
                                borderRadius: '0.5rem',
                                fontSize: '0.875rem',
                                lineHeight: '1.6',
                                maxHeight: '200px',
                                overflowY: 'auto'
                            }}>
                                {result.llm_explanation_patient || 'Your brain health assessment is complete. Please discuss these results with your doctor.'}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Recommendations */}
                <div className="card" style={{ gridColumn: 'span 2' }}>
                    <div className="card-header">
                        <h3 className="card-title">Recommendations</h3>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
                        {getRecommendations(result.predicted_class).map((rec, idx) => (
                            <div key={idx} style={{
                                display: 'flex',
                                alignItems: 'flex-start',
                                gap: '0.75rem',
                                padding: '0.75rem',
                                background: 'var(--lighter)',
                                borderRadius: '0.5rem'
                            }}>
                                <CheckCircle size={18} color="var(--primary)" style={{ flexShrink: 0, marginTop: '2px' }} />
                                <span style={{ fontSize: '0.875rem' }}>{rec}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}

function getRecommendations(predictedClass) {
    const recommendations = {
        'NonDemented': [
            'Continue routine cognitive health monitoring',
            'Maintain cardiovascular health and physical activity',
            'Engage in cognitively stimulating activities',
            'Follow up as per standard care protocols'
        ],
        'VeryMildDemented': [
            'Schedule comprehensive neuropsychological evaluation',
            'Consider referral to memory clinic',
            'Evaluate for reversible causes of cognitive decline',
            'Discuss cognitive rehabilitation options',
            'Follow up in 6 months for repeat assessment'
        ],
        'MildDemented': [
            'Urgent referral to neurologist/geriatrician',
            'Consider pharmacological treatment options',
            'Initiate care planning discussions',
            'Assess functional abilities and safety',
            'Connect with caregiver support resources'
        ],
        'ModerateDemented': [
            'Immediate specialist consultation required',
            'Comprehensive safety assessment needed',
            'Structured care plan development',
            'Caregiver education and support',
            'Coordinate with social services'
        ]
    }
    return recommendations[predictedClass] || ['Consult with healthcare provider']
}

export default Results

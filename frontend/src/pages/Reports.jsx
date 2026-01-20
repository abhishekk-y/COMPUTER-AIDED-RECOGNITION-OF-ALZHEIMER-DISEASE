import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { FileText, Download, Eye, Calendar, Search, Filter, RefreshCw, Plus } from 'lucide-react'
import api from '../services/api'

function Reports() {
    const navigate = useNavigate()
    const [searchTerm, setSearchTerm] = useState('')
    const [reports, setReports] = useState([])
    const [predictions, setPredictions] = useState([])
    const [loading, setLoading] = useState(true)
    const [generating, setGenerating] = useState(null)

    useEffect(() => {
        fetchData()
    }, [])

    const fetchData = async () => {
        setLoading(true)
        try {
            // Fetch existing reports
            const reportsRes = await api.get('/api/reports/')
            setReports(reportsRes.data)

            // Fetch predictions without reports
            const predsRes = await api.get('/api/predictions/')
            const withoutReports = predsRes.data.filter(p => !p.report_path)
            setPredictions(withoutReports)
        } catch (err) {
            console.error('Failed to fetch reports:', err)
        } finally {
            setLoading(false)
        }
    }

    const handleGenerateReport = async (predictionId) => {
        setGenerating(predictionId)
        try {
            const res = await api.post('/api/reports/generate', {
                prediction_id: predictionId,
                include_gradcam: true,
                include_heatmap: true,
                include_explanation: true
            })

            // Refresh data
            await fetchData()

            // Open download
            window.open(`http://localhost:8000${res.data.download_url}`, '_blank')
        } catch (err) {
            console.error('Report generation failed:', err)
            alert('Failed to generate report')
        } finally {
            setGenerating(null)
        }
    }

    const handleDownload = (downloadUrl) => {
        window.open(`http://localhost:8000${downloadUrl}`, '_blank')
    }

    const getResultBadge = (result) => {
        const badges = {
            'NonDemented': 'badge-success',
            'VeryMildDemented': 'badge-warning',
            'MildDemented': 'badge-danger',
            'ModerateDemented': 'badge-danger'
        }
        return badges[result] || 'badge-info'
    }

    const filteredReports = reports.filter(r =>
        r.patient_id?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        r.patient_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        r.predicted_class?.toLowerCase().includes(searchTerm.toLowerCase())
    )

    if (loading) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
                <div className="spinner" />
            </div>
        )
    }

    return (
        <div>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <div style={{ position: 'relative' }}>
                        <Search size={18} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--gray-light)' }} />
                        <input
                            type="text"
                            className="form-input"
                            placeholder="Search reports..."
                            style={{ paddingLeft: '40px', width: '300px' }}
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                        />
                    </div>
                    <button className="btn btn-secondary" onClick={fetchData}>
                        <RefreshCw size={18} />
                    </button>
                </div>
                <button className="btn btn-primary" onClick={() => navigate('/predict')}>
                    <Plus size={18} /> New Prediction
                </button>
            </div>

            {/* Pending Reports Section */}
            {predictions.length > 0 && (
                <div className="card" style={{ marginBottom: '1.5rem' }}>
                    <div className="card-header">
                        <h3 className="card-title">Predictions Without Reports</h3>
                        <span className="badge badge-warning">{predictions.length} pending</span>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1rem' }}>
                        {predictions.slice(0, 6).map(pred => (
                            <div key={pred.id} style={{
                                padding: '1rem',
                                background: 'var(--lighter)',
                                borderRadius: '0.5rem',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center'
                            }}>
                                <div>
                                    <span className={`badge ${getResultBadge(pred.predicted_class)}`} style={{ marginBottom: '0.5rem' }}>
                                        {pred.predicted_class}
                                    </span>
                                    <p style={{ fontSize: '0.875rem', margin: 0 }}>
                                        Patient #{pred.patient_id}
                                    </p>
                                </div>
                                <button
                                    className="btn btn-primary"
                                    onClick={() => handleGenerateReport(pred.id)}
                                    disabled={generating === pred.id}
                                    style={{ padding: '0.5rem 1rem' }}
                                >
                                    {generating === pred.id ? (
                                        <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} />
                                    ) : (
                                        <><FileText size={16} /> Generate</>
                                    )}
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Generated Reports Table */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Generated Reports</h3>
                    <span style={{ fontSize: '0.875rem', color: 'var(--gray)' }}>
                        {filteredReports.length} reports
                    </span>
                </div>

                {filteredReports.length > 0 ? (
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '2px solid var(--light)' }}>
                                <th style={{ textAlign: 'left', padding: '1rem', fontSize: '0.75rem', color: 'var(--gray)', fontWeight: '600', textTransform: 'uppercase' }}>Patient</th>
                                <th style={{ textAlign: 'left', padding: '1rem', fontSize: '0.75rem', color: 'var(--gray)', fontWeight: '600', textTransform: 'uppercase' }}>Result</th>
                                <th style={{ textAlign: 'left', padding: '1rem', fontSize: '0.75rem', color: 'var(--gray)', fontWeight: '600', textTransform: 'uppercase' }}>Date</th>
                                <th style={{ textAlign: 'right', padding: '1rem', fontSize: '0.75rem', color: 'var(--gray)', fontWeight: '600', textTransform: 'uppercase' }}>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredReports.map(report => (
                                <tr key={report.id} style={{ borderBottom: '1px solid var(--light)' }}>
                                    <td style={{ padding: '1rem' }}>
                                        <div>
                                            <span style={{ fontWeight: '500' }}>{report.patient_name}</span>
                                            <br />
                                            <span style={{ fontSize: '0.75rem', color: 'var(--gray)' }}>{report.patient_id}</span>
                                        </div>
                                    </td>
                                    <td style={{ padding: '1rem' }}>
                                        <span className={`badge ${getResultBadge(report.predicted_class)}`}>
                                            {report.predicted_class}
                                        </span>
                                    </td>
                                    <td style={{ padding: '1rem', fontSize: '0.875rem', color: 'var(--gray)' }}>
                                        <Calendar size={14} style={{ marginRight: '0.25rem', verticalAlign: 'middle' }} />
                                        {new Date(report.created_at).toLocaleDateString()}
                                    </td>
                                    <td style={{ padding: '1rem', textAlign: 'right' }}>
                                        <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
                                            <button
                                                className="btn btn-icon btn-secondary"
                                                title="View Results"
                                                onClick={() => navigate(`/results/${report.id}`)}
                                            >
                                                <Eye size={16} />
                                            </button>
                                            <button
                                                className="btn btn-icon btn-primary"
                                                title="Download PDF"
                                                onClick={() => handleDownload(report.download_url)}
                                            >
                                                <Download size={16} />
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                ) : (
                    <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--gray)' }}>
                        <FileText size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
                        <p>No reports found</p>
                        <button className="btn btn-primary" onClick={() => navigate('/predict')}>
                            Create First Prediction
                        </button>
                    </div>
                )}
            </div>
        </div>
    )
}

export default Reports

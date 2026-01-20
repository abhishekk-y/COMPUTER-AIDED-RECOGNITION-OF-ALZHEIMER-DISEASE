import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
    Brain,
    Users,
    FileCheck,
    Activity,
    TrendingUp,
    Clock,
    RefreshCw
} from 'lucide-react'
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, AreaChart, Area } from 'recharts'
import api from '../services/api'

function Dashboard() {
    const navigate = useNavigate()
    const [loading, setLoading] = useState(true)
    const [stats, setStats] = useState({
        totalPatients: 0,
        totalPredictions: 0,
        classDistribution: {},
        recentPredictions: []
    })

    const [modelMetrics, setModelMetrics] = useState({
        accuracy: 0,
        precision: 0,
        recall: 0,
        f1Score: 0
    })

    const fetchData = async () => {
        setLoading(true)
        try {
            // Fetch prediction stats
            const statsRes = await api.get('/api/predictions/stats/summary')

            // Fetch patients count
            const patientsRes = await api.get('/api/patients/')

            // Fetch model metrics
            let metrics = { accuracy: 0, precision: 0, recall: 0, f1Score: 0 }
            try {
                const metricsRes = await api.get('/api/admin/model-metrics')
                metrics = metricsRes.data
            } catch (e) {
                // Use defaults if no metrics available
            }

            setStats({
                totalPatients: patientsRes.data.length,
                totalPredictions: statsRes.data.total_predictions,
                classDistribution: statsRes.data.class_distribution || {},
                recentPredictions: statsRes.data.recent_predictions || []
            })

            setModelMetrics(metrics)
        } catch (err) {
            console.error('Failed to fetch dashboard data:', err)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchData()
        // Auto-refresh every 30 seconds
        const interval = setInterval(fetchData, 30000)
        return () => clearInterval(interval)
    }, [])

    const pieData = Object.entries(stats.classDistribution).map(([name, value]) => ({
        name,
        value,
        color: {
            'NonDemented': '#10B981',
            'VeryMildDemented': '#F59E0B',
            'MildDemented': '#F97316',
            'ModerateDemented': '#EF4444'
        }[name] || '#6B7280'
    }))

    const getResultBadge = (result) => {
        const badges = {
            'NonDemented': 'badge-success',
            'VeryMildDemented': 'badge-warning',
            'MildDemented': 'badge-danger',
            'ModerateDemented': 'badge-danger'
        }
        return badges[result] || 'badge-info'
    }

    if (loading && stats.totalPredictions === 0) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
                <div className="spinner" />
            </div>
        )
    }

    return (
        <div>
            {/* Header with refresh */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <h2>Dashboard Overview</h2>
                <button className="btn btn-secondary" onClick={fetchData} disabled={loading}>
                    <RefreshCw size={16} className={loading ? 'spinning' : ''} />
                    Refresh
                </button>
            </div>

            {/* Stats Grid */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon primary">
                        <Users size={24} />
                    </div>
                    <div className="stat-content">
                        <h3>{stats.totalPatients}</h3>
                        <p>Total Patients</p>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon success">
                        <FileCheck size={24} />
                    </div>
                    <div className="stat-content">
                        <h3>{stats.totalPredictions}</h3>
                        <p>Predictions Made</p>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon warning">
                        <TrendingUp size={24} />
                    </div>
                    <div className="stat-content">
                        <h3>{modelMetrics.accuracy ? `${(modelMetrics.accuracy * 100).toFixed(1)}%` : 'N/A'}</h3>
                        <p>Model Accuracy</p>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon danger">
                        <Activity size={24} />
                    </div>
                    <div className="stat-content">
                        <h3>Active</h3>
                        <p>System Status</p>
                    </div>
                </div>
            </div>

            {/* Charts Row */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
                {/* Prediction Distribution */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Prediction Distribution</h3>
                    </div>
                    {pieData.length > 0 ? (
                        <div style={{ height: '250px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={pieData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={60}
                                        outerRadius={90}
                                        paddingAngle={5}
                                        dataKey="value"
                                        label={({ name, percent }) => `${name.replace('Demented', '')} ${(percent * 100).toFixed(0)}%`}
                                        labelLine={false}
                                    >
                                        {pieData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <div style={{ height: '250px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--gray)' }}>
                            No predictions yet
                        </div>
                    )}
                </div>

                {/* Model Performance */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Model Performance</h3>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', padding: '1rem' }}>
                        {[
                            { label: 'Accuracy', value: modelMetrics.accuracy, color: '#10B981' },
                            { label: 'Precision', value: modelMetrics.precision, color: '#4F46E5' },
                            { label: 'Recall', value: modelMetrics.recall, color: '#06B6D4' },
                            { label: 'F1 Score', value: modelMetrics.f1Score, color: '#F59E0B' }
                        ].map(metric => (
                            <div key={metric.label} style={{ textAlign: 'center' }}>
                                <div style={{
                                    width: '70px',
                                    height: '70px',
                                    borderRadius: '50%',
                                    border: `4px solid ${metric.color}`,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    margin: '0 auto 0.5rem',
                                    fontWeight: '700'
                                }}>
                                    {metric.value ? `${(metric.value * 100).toFixed(0)}%` : 'N/A'}
                                </div>
                                <p style={{ fontSize: '0.875rem', color: 'var(--gray)', margin: 0 }}>{metric.label}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Quick Actions & Recent */}
            <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '1.5rem' }}>
                {/* Quick Actions */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Quick Actions</h3>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        <button
                            className="btn btn-primary btn-lg"
                            onClick={() => navigate('/predict')}
                            style={{ width: '100%', justifyContent: 'flex-start' }}
                        >
                            <Brain size={20} />
                            New Prediction
                        </button>
                        <button
                            className="btn btn-secondary"
                            onClick={() => navigate('/chat')}
                            style={{ width: '100%', justifyContent: 'flex-start' }}
                        >
                            <Activity size={20} />
                            AI Assistant
                        </button>
                        <button
                            className="btn btn-secondary"
                            onClick={() => navigate('/reports')}
                            style={{ width: '100%', justifyContent: 'flex-start' }}
                        >
                            <FileCheck size={20} />
                            View Reports
                        </button>
                    </div>
                </div>

                {/* Recent Predictions */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Recent Predictions</h3>
                        <span style={{ fontSize: '0.75rem', color: 'var(--gray)' }}>
                            Real-time updates
                        </span>
                    </div>
                    {stats.recentPredictions.length > 0 ? (
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                            <thead>
                                <tr style={{ borderBottom: '1px solid var(--light)' }}>
                                    <th style={{ textAlign: 'left', padding: '0.75rem', fontSize: '0.75rem', color: 'var(--gray)', fontWeight: '500' }}>Patient</th>
                                    <th style={{ textAlign: 'left', padding: '0.75rem', fontSize: '0.75rem', color: 'var(--gray)', fontWeight: '500' }}>Result</th>
                                    <th style={{ textAlign: 'left', padding: '0.75rem', fontSize: '0.75rem', color: 'var(--gray)', fontWeight: '500' }}>Confidence</th>
                                    <th style={{ textAlign: 'left', padding: '0.75rem', fontSize: '0.75rem', color: 'var(--gray)', fontWeight: '500' }}>Time</th>
                                </tr>
                            </thead>
                            <tbody>
                                {stats.recentPredictions.map(pred => (
                                    <tr
                                        key={pred.id}
                                        style={{ borderBottom: '1px solid var(--light)', cursor: 'pointer' }}
                                        onClick={() => navigate(`/results/${pred.id}`)}
                                    >
                                        <td style={{ padding: '0.75rem' }}>
                                            <div>
                                                <span style={{ fontWeight: '500', fontSize: '0.875rem' }}>{pred.patient_name}</span>
                                                <br />
                                                <span style={{ fontSize: '0.75rem', color: 'var(--gray)' }}>{pred.patient_id}</span>
                                            </div>
                                        </td>
                                        <td style={{ padding: '0.75rem' }}>
                                            <span className={`badge ${getResultBadge(pred.predicted_class)}`}>
                                                {pred.predicted_class.replace('Demented', '')}
                                            </span>
                                        </td>
                                        <td style={{ padding: '0.75rem', fontSize: '0.875rem' }}>
                                            {(pred.confidence * 100).toFixed(1)}%
                                        </td>
                                        <td style={{ padding: '0.75rem', fontSize: '0.875rem', color: 'var(--gray)' }}>
                                            <Clock size={14} style={{ marginRight: '0.25rem', verticalAlign: 'middle' }} />
                                            {new Date(pred.created_at).toLocaleDateString()}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : (
                        <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--gray)' }}>
                            <Brain size={48} style={{ marginBottom: '1rem', opacity: 0.3 }} />
                            <p>No predictions yet. Run your first analysis!</p>
                            <button className="btn btn-primary" onClick={() => navigate('/predict')}>
                                New Prediction
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default Dashboard

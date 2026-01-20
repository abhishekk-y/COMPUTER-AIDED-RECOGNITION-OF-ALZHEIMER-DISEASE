import React, { useState, useEffect } from 'react'
import {
    Database,
    RefreshCw,
    TrendingUp,
    Upload,
    Play,
    CheckCircle,
    BarChart3,
    Brain,
    Activity
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, Cell } from 'recharts'
import api from '../services/api'

function Admin() {
    const [loading, setLoading] = useState(true)
    const [training, setTraining] = useState(false)
    const [trainingProgress, setTrainingProgress] = useState(0)
    const [trainingStatus, setTrainingStatus] = useState(null)

    const [datasetStats, setDatasetStats] = useState({
        totalImages: 0,
        classes: {},
        trainSize: 0,
        valSize: 0,
        testSize: 0
    })

    const [modelMetrics, setModelMetrics] = useState({
        accuracy: 0,
        precision: 0,
        recall: 0,
        f1Score: 0,
        auc: 0,
        confusionMatrix: [],
        trainingHistory: []
    })

    useEffect(() => {
        fetchData()
    }, [])

    const fetchData = async () => {
        setLoading(true)
        try {
            // Fetch dashboard stats
            const statsRes = await api.get('/api/admin/dashboard')
            setDatasetStats(prev => ({
                ...prev,
                totalImages: statsRes.data.dataset_images || 0,
                classes: statsRes.data.class_distribution || {}
            }))

            // Fetch model metrics
            try {
                const metricsRes = await api.get('/api/admin/model-metrics')
                setModelMetrics({
                    accuracy: metricsRes.data.accuracy || 0,
                    precision: metricsRes.data.precision || 0,
                    recall: metricsRes.data.recall || 0,
                    f1Score: metricsRes.data.f1_score || 0,
                    auc: metricsRes.data.auc_roc || 0,
                    confusionMatrix: metricsRes.data.confusion_matrix || [],
                    trainingHistory: metricsRes.data.training_history || []
                })
            } catch (e) {
                // Use defaults if no model trained yet
            }

            // Check training status
            try {
                const statusRes = await api.get('/api/admin/training-status')
                setTrainingStatus(statusRes.data)
                if (statusRes.data.status === 'training') {
                    setTraining(true)
                    setTrainingProgress(statusRes.data.progress_percent || 0)
                }
            } catch (e) { }

        } catch (err) {
            console.error('Failed to fetch admin data:', err)
        } finally {
            setLoading(false)
        }
    }

    const handleRetrain = async () => {
        setTraining(true)
        setTrainingProgress(0)

        try {
            await api.post('/api/admin/train', {
                epochs: 50,
                batch_size: 32,
                learning_rate: 0.001
            })

            // Poll for status
            const pollInterval = setInterval(async () => {
                try {
                    const res = await api.get('/api/admin/training-status')
                    setTrainingProgress(res.data.progress_percent || 0)

                    if (res.data.status === 'completed' || res.data.status === 'failed') {
                        clearInterval(pollInterval)
                        setTraining(false)
                        await fetchData()
                    }
                } catch (e) { }
            }, 2000)

        } catch (err) {
            console.error('Training failed:', err)
            setTraining(false)
            alert('Failed to start training. Check backend logs.')
        }
    }

    const classColors = {
        'NonDemented': '#10B981',
        'VeryMildDemented': '#F59E0B',
        'MildDemented': '#F97316',
        'ModerateDemented': '#EF4444'
    }

    // Prepare chart data
    const classData = Object.entries(datasetStats.classes).map(([name, count]) => ({
        name: name.replace('Demented', ''),
        fullName: name,
        count,
        color: classColors[name] || '#6B7280'
    }))

    const totalImages = classData.reduce((sum, d) => sum + d.count, 0)

    if (loading) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
                <div className="spinner" />
            </div>
        )
    }

    return (
        <div>
            {/* Stats Grid */}
            <div className="stats-grid" style={{ marginBottom: '1.5rem' }}>
                <div className="stat-card">
                    <div className="stat-icon primary">
                        <Database size={24} />
                    </div>
                    <div className="stat-content">
                        <h3>{totalImages.toLocaleString()}</h3>
                        <p>Dataset Images</p>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon success">
                        <TrendingUp size={24} />
                    </div>
                    <div className="stat-content">
                        <h3>{modelMetrics.accuracy ? `${(modelMetrics.accuracy * 100).toFixed(1)}%` : 'N/A'}</h3>
                        <p>Model Accuracy</p>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon warning">
                        <BarChart3 size={24} />
                    </div>
                    <div className="stat-content">
                        <h3>{modelMetrics.f1Score ? `${(modelMetrics.f1Score * 100).toFixed(1)}%` : 'N/A'}</h3>
                        <p>F1 Score</p>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon danger">
                        <Activity size={24} />
                    </div>
                    <div className="stat-content">
                        <h3>{training ? 'Training' : 'Ready'}</h3>
                        <p>Model Status</p>
                    </div>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                {/* Dataset Distribution */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Dataset Distribution</h3>
                        <button className="btn btn-secondary" onClick={fetchData} style={{ fontSize: '0.75rem' }}>
                            <RefreshCw size={14} /> Refresh
                        </button>
                    </div>

                    {classData.length > 0 ? (
                        <>
                            <div style={{ height: '200px', marginBottom: '1rem' }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={classData}>
                                        <XAxis dataKey="name" />
                                        <YAxis />
                                        <Tooltip formatter={(v) => v.toLocaleString()} />
                                        <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                                            {classData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.color} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>

                            {classData.map(cls => (
                                <div key={cls.fullName} style={{ marginBottom: '0.75rem' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                        <span style={{ fontSize: '0.875rem' }}>{cls.fullName}</span>
                                        <span style={{ fontSize: '0.875rem', color: 'var(--gray)' }}>
                                            {cls.count.toLocaleString()} ({((cls.count / totalImages) * 100).toFixed(1)}%)
                                        </span>
                                    </div>
                                    <div style={{ height: '6px', background: 'var(--light)', borderRadius: '3px' }}>
                                        <div style={{
                                            width: `${(cls.count / totalImages) * 100}%`,
                                            height: '100%',
                                            background: cls.color,
                                            borderRadius: '3px'
                                        }} />
                                    </div>
                                </div>
                            ))}
                        </>
                    ) : (
                        <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--gray)' }}>
                            <Database size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                            <p>No dataset statistics available</p>
                        </div>
                    )}
                </div>

                {/* Model Performance */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Model Performance</h3>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                        {[
                            { label: 'Accuracy', value: modelMetrics.accuracy, color: '#10B981' },
                            { label: 'Precision', value: modelMetrics.precision, color: '#4F46E5' },
                            { label: 'Recall', value: modelMetrics.recall, color: '#06B6D4' },
                            { label: 'F1 Score', value: modelMetrics.f1Score, color: '#F59E0B' },
                        ].map(metric => (
                            <div key={metric.label} style={{ textAlign: 'center', padding: '1rem', background: 'var(--lighter)', borderRadius: '0.5rem' }}>
                                <div style={{
                                    width: '70px',
                                    height: '70px',
                                    borderRadius: '50%',
                                    border: `4px solid ${metric.color}`,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    margin: '0 auto 0.5rem',
                                    fontWeight: '700',
                                    fontSize: '1rem'
                                }}>
                                    {metric.value ? `${(metric.value * 100).toFixed(0)}%` : 'N/A'}
                                </div>
                                <p style={{ fontSize: '0.875rem', color: 'var(--gray)', margin: 0 }}>{metric.label}</p>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Training Controls */}
                <div className="card" style={{ gridColumn: 'span 2' }}>
                    <div className="card-header">
                        <h3 className="card-title">Model Training</h3>
                        <button
                            className="btn btn-primary"
                            onClick={handleRetrain}
                            disabled={training}
                        >
                            {training ? (
                                <>
                                    <RefreshCw size={16} style={{ animation: 'spin 1s linear infinite' }} />
                                    Training... {trainingProgress.toFixed(0)}%
                                </>
                            ) : (
                                <>
                                    <Play size={16} /> Start Training
                                </>
                            )}
                        </button>
                    </div>

                    {training && (
                        <div style={{ marginBottom: '1.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                <span>Training Progress</span>
                                <span>{trainingProgress.toFixed(1)}%</span>
                            </div>
                            <div style={{ height: '12px', background: 'var(--light)', borderRadius: '6px' }}>
                                <div style={{
                                    width: `${trainingProgress}%`,
                                    height: '100%',
                                    background: 'var(--gradient-primary)',
                                    borderRadius: '6px',
                                    transition: 'width 0.3s'
                                }} />
                            </div>
                        </div>
                    )}

                    {/* Training History Chart */}
                    {modelMetrics.trainingHistory && modelMetrics.trainingHistory.length > 0 ? (
                        <div style={{ height: '250px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={modelMetrics.trainingHistory}>
                                    <defs>
                                        <linearGradient id="trainGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#4F46E5" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#4F46E5" stopOpacity={0} />
                                        </linearGradient>
                                        <linearGradient id="valGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#10B981" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <XAxis dataKey="epoch" />
                                    <YAxis domain={[0, 100]} />
                                    <Tooltip />
                                    <Area type="monotone" dataKey="train_acc" stroke="#4F46E5" fill="url(#trainGradient)" name="Training" />
                                    <Area type="monotone" dataKey="val_acc" stroke="#10B981" fill="url(#valGradient)" name="Validation" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--gray)' }}>
                            <Brain size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                            <p>No training history available</p>
                            <p style={{ fontSize: '0.875rem' }}>Train a model to see accuracy curves</p>
                        </div>
                    )}

                    <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', marginTop: '1rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#4F46E5' }} />
                            <span style={{ fontSize: '0.875rem' }}>Training Accuracy</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#10B981' }} />
                            <span style={{ fontSize: '0.875rem' }}>Validation Accuracy</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Admin

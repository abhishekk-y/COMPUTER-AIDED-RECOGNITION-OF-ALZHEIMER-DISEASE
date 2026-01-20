import React, { useState } from 'react'
import { Brain, Mail, Lock, User, Eye, EyeOff } from 'lucide-react'
import api from '../services/api'

function Login({ onLogin }) {
    const [isRegister, setIsRegister] = useState(false)
    const [showPassword, setShowPassword] = useState(false)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const [formData, setFormData] = useState({
        username: '',
        email: '',
        password: '',
        full_name: ''
    })

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value })
        setError('')
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        setLoading(true)
        setError('')

        try {
            if (isRegister) {
                // Register
                await api.post('/api/auth/register', {
                    username: formData.username,
                    email: formData.email,
                    password: formData.password,
                    full_name: formData.full_name,
                    role: 'clinician'
                })
                // Then login
                const loginRes = await api.post('/api/auth/login',
                    new URLSearchParams({
                        username: formData.username,
                        password: formData.password
                    }),
                    { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
                )
                onLogin({ username: formData.username, full_name: formData.full_name }, loginRes.data.access_token)
            } else {
                // Login
                const response = await api.post('/api/auth/login',
                    new URLSearchParams({
                        username: formData.username,
                        password: formData.password
                    }),
                    { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
                )
                onLogin({ username: formData.username }, response.data.access_token)
            }
        } catch (err) {
            setError(err.response?.data?.detail || 'Authentication failed. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    // Demo login (skip auth for development)
    const handleDemoLogin = () => {
        onLogin({ username: 'demo', full_name: 'Demo User', role: 'admin' }, 'demo-token')
    }

    return (
        <div style={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'linear-gradient(135deg, #1F2937 0%, #374151 50%, #1F2937 100%)',
            padding: '2rem'
        }}>
            <div style={{
                width: '100%',
                maxWidth: '420px',
                background: 'white',
                borderRadius: '1rem',
                boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
                overflow: 'hidden'
            }}>
                {/* Header */}
                <div style={{
                    background: 'linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%)',
                    padding: '2rem',
                    textAlign: 'center'
                }}>
                    <div style={{
                        width: '80px',
                        height: '80px',
                        background: 'rgba(255,255,255,0.2)',
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        margin: '0 auto 1rem'
                    }}>
                        <Brain size={44} color="white" />
                    </div>
                    <h1 style={{ color: 'white', fontSize: '1.75rem', marginBottom: '0.5rem' }}>
                        CARE-AD+
                    </h1>
                    <p style={{ color: 'rgba(255,255,255,0.8)', fontSize: '0.875rem' }}>
                        Computer-Aided Alzheimer's Detection
                    </p>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} style={{ padding: '2rem' }}>
                    <h2 style={{
                        fontSize: '1.25rem',
                        marginBottom: '1.5rem',
                        textAlign: 'center',
                        color: 'var(--dark)'
                    }}>
                        {isRegister ? 'Create Account' : 'Welcome Back'}
                    </h2>

                    {error && (
                        <div style={{
                            background: 'rgba(239, 68, 68, 0.1)',
                            color: 'var(--danger)',
                            padding: '0.75rem',
                            borderRadius: '0.5rem',
                            marginBottom: '1rem',
                            fontSize: '0.875rem'
                        }}>
                            {error}
                        </div>
                    )}

                    {isRegister && (
                        <div className="form-group">
                            <label className="form-label">Full Name</label>
                            <div style={{ position: 'relative' }}>
                                <User size={18} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--gray-light)' }} />
                                <input
                                    type="text"
                                    name="full_name"
                                    className="form-input"
                                    style={{ paddingLeft: '40px' }}
                                    placeholder="Dr. John Smith"
                                    value={formData.full_name}
                                    onChange={handleChange}
                                    required={isRegister}
                                />
                            </div>
                        </div>
                    )}

                    <div className="form-group">
                        <label className="form-label">Username</label>
                        <div style={{ position: 'relative' }}>
                            <User size={18} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--gray-light)' }} />
                            <input
                                type="text"
                                name="username"
                                className="form-input"
                                style={{ paddingLeft: '40px' }}
                                placeholder="username"
                                value={formData.username}
                                onChange={handleChange}
                                required
                            />
                        </div>
                    </div>

                    {isRegister && (
                        <div className="form-group">
                            <label className="form-label">Email</label>
                            <div style={{ position: 'relative' }}>
                                <Mail size={18} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--gray-light)' }} />
                                <input
                                    type="email"
                                    name="email"
                                    className="form-input"
                                    style={{ paddingLeft: '40px' }}
                                    placeholder="doctor@hospital.com"
                                    value={formData.email}
                                    onChange={handleChange}
                                    required={isRegister}
                                />
                            </div>
                        </div>
                    )}

                    <div className="form-group">
                        <label className="form-label">Password</label>
                        <div style={{ position: 'relative' }}>
                            <Lock size={18} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--gray-light)' }} />
                            <input
                                type={showPassword ? 'text' : 'password'}
                                name="password"
                                className="form-input"
                                style={{ paddingLeft: '40px', paddingRight: '40px' }}
                                placeholder="••••••••"
                                value={formData.password}
                                onChange={handleChange}
                                required
                            />
                            <button
                                type="button"
                                onClick={() => setShowPassword(!showPassword)}
                                style={{
                                    position: 'absolute',
                                    right: '12px',
                                    top: '50%',
                                    transform: 'translateY(-50%)',
                                    background: 'none',
                                    border: 'none',
                                    cursor: 'pointer',
                                    color: 'var(--gray-light)'
                                }}
                            >
                                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                            </button>
                        </div>
                    </div>

                    <button
                        type="submit"
                        className="btn btn-primary btn-lg"
                        style={{ width: '100%', marginTop: '0.5rem' }}
                        disabled={loading}
                    >
                        {loading ? (
                            <div className="spinner" style={{ width: '20px', height: '20px', borderWidth: '2px' }} />
                        ) : (
                            isRegister ? 'Create Account' : 'Sign In'
                        )}
                    </button>

                    <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
                        <button
                            type="button"
                            onClick={() => setIsRegister(!isRegister)}
                            style={{
                                background: 'none',
                                border: 'none',
                                color: 'var(--primary)',
                                cursor: 'pointer',
                                fontSize: '0.875rem'
                            }}
                        >
                            {isRegister
                                ? 'Already have an account? Sign in'
                                : "Don't have an account? Register"}
                        </button>
                    </div>

                    {/* Demo Login */}
                    <div style={{
                        marginTop: '1.5rem',
                        paddingTop: '1.5rem',
                        borderTop: '1px solid var(--light)',
                        textAlign: 'center'
                    }}>
                        <button
                            type="button"
                            onClick={handleDemoLogin}
                            className="btn btn-secondary"
                            style={{ width: '100%' }}
                        >
                            🚀 Try Demo (Skip Login)
                        </button>
                    </div>
                </form>
            </div>
        </div>
    )
}

export default Login

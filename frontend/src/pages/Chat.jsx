import React, { useState, useRef, useEffect } from 'react'
import { Send, Bot, Sparkles, RefreshCw, Trash2 } from 'lucide-react'
import api from '../services/api'

function Chat() {
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content: `Hello! I'm the CARE-AD+ AI Assistant. I can help you understand Alzheimer's disease prediction results, explain brain imaging findings, and answer questions about cognitive assessments.

How can I assist you today?`
        }
    ])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [mode, setMode] = useState('technical')
    const [predictionId, setPredictionId] = useState(null)
    const [recentPredictions, setRecentPredictions] = useState([])
    const messagesEndRef = useRef(null)

    useEffect(() => {
        fetchRecentPredictions()
    }, [])

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    const fetchRecentPredictions = async () => {
        try {
            const res = await api.get('/api/predictions/?limit=5')
            setRecentPredictions(res.data)
            if (res.data.length > 0 && !predictionId) {
                setPredictionId(res.data[0].id)
            }
        } catch (err) {
            console.error('Failed to fetch predictions:', err)
        }
    }

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    const handleSend = async () => {
        if (!input.trim() || loading) return

        const userMessage = input.trim()
        setInput('')
        setMessages(prev => [...prev, { role: 'user', content: userMessage }])
        setLoading(true)

        try {
            const res = await api.post('/api/chat/', {
                message: userMessage,
                prediction_id: predictionId,
                mode: mode
            })

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: res.data.response
            }])
        } catch (err) {
            console.error('Chat error:', err)
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'I apologize, but I encountered an error processing your request. Please ensure the backend server is running and the LLM service is available.'
            }])
        } finally {
            setLoading(false)
        }
    }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSend()
        }
    }

    const clearChat = () => {
        setMessages([{
            role: 'assistant',
            content: 'Chat cleared. How can I help you?'
        }])
    }

    return (
        <div style={{ maxWidth: '900px', margin: '0 auto' }}>
            {/* Controls */}
            <div className="card" style={{ marginBottom: '1rem', padding: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
                    {/* Mode Toggle */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Sparkles size={20} color="var(--primary)" />
                        <span style={{ fontWeight: '500' }}>Mode:</span>
                        <div style={{ display: 'flex', gap: '0.25rem' }}>
                            <button
                                className={`btn ${mode === 'technical' ? 'btn-primary' : 'btn-secondary'}`}
                                onClick={() => setMode('technical')}
                                style={{ padding: '0.25rem 0.75rem' }}
                            >
                                Technical
                            </button>
                            <button
                                className={`btn ${mode === 'patient' ? 'btn-primary' : 'btn-secondary'}`}
                                onClick={() => setMode('patient')}
                                style={{ padding: '0.25rem 0.75rem' }}
                            >
                                Simple
                            </button>
                        </div>
                    </div>

                    {/* Context Selector */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ fontSize: '0.875rem', color: 'var(--gray)' }}>Context:</span>
                        <select
                            className="form-input"
                            style={{ width: 'auto', padding: '0.25rem 0.5rem' }}
                            value={predictionId || ''}
                            onChange={(e) => setPredictionId(e.target.value ? parseInt(e.target.value) : null)}
                        >
                            <option value="">No context</option>
                            {recentPredictions.map(p => (
                                <option key={p.id} value={p.id}>
                                    Prediction #{p.id} - {p.predicted_class}
                                </option>
                            ))}
                        </select>
                    </div>

                    <button className="btn btn-secondary" onClick={clearChat}>
                        <Trash2 size={16} /> Clear
                    </button>
                </div>
            </div>

            {/* Chat Container */}
            <div className="chat-container" style={{ height: '500px' }}>
                <div className="chat-messages">
                    {messages.map((msg, idx) => (
                        <div key={idx} className={`chat-message ${msg.role}`}>
                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                                {msg.role === 'assistant' && (
                                    <div style={{
                                        width: '28px',
                                        height: '28px',
                                        borderRadius: '50%',
                                        background: 'var(--gradient-primary)',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        flexShrink: 0
                                    }}>
                                        <Bot size={16} color="white" />
                                    </div>
                                )}
                                <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                            </div>
                        </div>
                    ))}
                    {loading && (
                        <div className="chat-message assistant">
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <RefreshCw size={16} style={{ animation: 'spin 1s linear infinite' }} />
                                Thinking...
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <div className="chat-input-area">
                    <input
                        type="text"
                        className="form-input"
                        placeholder="Ask about Alzheimer's disease, brain imaging, or test results..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={handleKeyPress}
                        disabled={loading}
                    />
                    <button
                        className="btn btn-primary"
                        onClick={handleSend}
                        disabled={loading || !input.trim()}
                    >
                        <Send size={18} />
                    </button>
                </div>
            </div>

            {/* Suggested Questions */}
            <div style={{ marginTop: '1rem' }}>
                <p style={{ fontSize: '0.875rem', color: 'var(--gray)', marginBottom: '0.5rem' }}>
                    Suggested questions:
                </p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {[
                        'What is Alzheimer\'s disease?',
                        'Explain the prediction result',
                        'What do these brain changes mean?',
                        'What are the next steps for this patient?'
                    ].map((q, idx) => (
                        <button
                            key={idx}
                            className="btn btn-secondary"
                            style={{ fontSize: '0.75rem', padding: '0.25rem 0.75rem' }}
                            onClick={() => setInput(q)}
                        >
                            {q}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    )
}

export default Chat

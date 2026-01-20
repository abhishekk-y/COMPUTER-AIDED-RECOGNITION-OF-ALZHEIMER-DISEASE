import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Prediction from './pages/Prediction'
import Results from './pages/Results'
import Chat from './pages/Chat'
import Reports from './pages/Reports'
import Admin from './pages/Admin'
import Login from './pages/Login'

function App() {
    const [isAuthenticated, setIsAuthenticated] = React.useState(false)
    const [user, setUser] = React.useState(null)

    // Check for existing token
    React.useEffect(() => {
        const token = localStorage.getItem('token')
        if (token) {
            setIsAuthenticated(true)
            // You could validate token here
        }
    }, [])

    const handleLogin = (userData, token) => {
        localStorage.setItem('token', token)
        setIsAuthenticated(true)
        setUser(userData)
    }

    const handleLogout = () => {
        localStorage.removeItem('token')
        setIsAuthenticated(false)
        setUser(null)
    }

    if (!isAuthenticated) {
        return <Login onLogin={handleLogin} />
    }

    return (
        <BrowserRouter>
            <Layout user={user} onLogout={handleLogout}>
                <Routes>
                    <Route path="/" element={<Dashboard />} />
                    <Route path="/predict" element={<Prediction />} />
                    <Route path="/results/:id" element={<Results />} />
                    <Route path="/chat" element={<Chat />} />
                    <Route path="/reports" element={<Reports />} />
                    <Route path="/admin" element={<Admin />} />
                    <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
            </Layout>
        </BrowserRouter>
    )
}

export default App

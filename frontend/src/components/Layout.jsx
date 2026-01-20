import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import {
    Brain,
    LayoutDashboard,
    Scan,
    MessageCircle,
    FileText,
    Settings,
    LogOut,
    Activity
} from 'lucide-react'

const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/predict', icon: Scan, label: 'New Prediction' },
    { path: '/chat', icon: MessageCircle, label: 'AI Assistant' },
    { path: '/reports', icon: FileText, label: 'Reports' },
    { path: '/admin', icon: Settings, label: 'Admin Panel' },
]

function Layout({ children, user, onLogout }) {
    const location = useLocation()

    return (
        <div className="app-layout">
            {/* Sidebar */}
            <aside className="sidebar">
                {/* Logo */}
                <div style={{ marginBottom: '2rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
                        <div style={{
                            width: '48px',
                            height: '48px',
                            background: 'linear-gradient(135deg, #4F46E5, #06B6D4)',
                            borderRadius: '12px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                        }}>
                            <Brain size={28} color="white" />
                        </div>
                        <div className="sidebar-text">
                            <h1 style={{ fontSize: '1.25rem', fontWeight: '700', margin: 0, color: 'white' }}>
                                CARE-AD+
                            </h1>
                            <p style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.6)', margin: 0 }}>
                                Alzheimer's Detection
                            </p>
                        </div>
                    </div>
                </div>

                {/* Navigation */}
                <nav>
                    {navItems.map(item => (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            style={({ isActive }) => ({
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.75rem',
                                padding: '0.75rem 1rem',
                                marginBottom: '0.5rem',
                                borderRadius: '0.5rem',
                                color: isActive ? 'white' : 'rgba(255,255,255,0.7)',
                                background: isActive ? 'rgba(255,255,255,0.1)' : 'transparent',
                                textDecoration: 'none',
                                transition: '0.2s ease',
                                fontWeight: isActive ? '500' : '400'
                            })}
                        >
                            <item.icon size={20} />
                            <span className="sidebar-text">{item.label}</span>
                        </NavLink>
                    ))}
                </nav>

                {/* User Info & Logout */}
                <div style={{
                    position: 'absolute',
                    bottom: '1.5rem',
                    left: '1.5rem',
                    right: '1.5rem',
                    paddingTop: '1rem',
                    borderTop: '1px solid rgba(255,255,255,0.1)'
                }}>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        marginBottom: '0.75rem'
                    }}>
                        <div className="sidebar-text">
                            <p style={{ fontSize: '0.875rem', fontWeight: '500', margin: 0, color: 'white' }}>
                                {user?.full_name || 'Clinician'}
                            </p>
                            <p style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.5)', margin: 0 }}>
                                {user?.role || 'User'}
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onLogout}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.5rem',
                            width: '100%',
                            padding: '0.5rem 0.75rem',
                            background: 'rgba(239, 68, 68, 0.2)',
                            color: '#F87171',
                            border: 'none',
                            borderRadius: '0.5rem',
                            cursor: 'pointer',
                            fontSize: '0.875rem',
                            transition: '0.2s'
                        }}
                    >
                        <LogOut size={18} />
                        <span className="sidebar-text">Logout</span>
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <main className="main-content">
                {/* Top Bar */}
                <header style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '2rem'
                }}>
                    <div>
                        <h2 style={{ fontSize: '1.5rem', marginBottom: '0.25rem' }}>
                            {navItems.find(item => item.path === location.pathname)?.label || 'CARE-AD+'}
                        </h2>
                        <p style={{ fontSize: '0.875rem', color: 'var(--gray)', margin: 0 }}>
                            {new Date().toLocaleDateString('en-US', {
                                weekday: 'long',
                                year: 'numeric',
                                month: 'long',
                                day: 'numeric'
                            })}
                        </p>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Activity size={16} color="var(--success)" />
                        <span style={{ fontSize: '0.875rem', color: 'var(--success)' }}>System Online</span>
                    </div>
                </header>

                {/* Page Content */}
                {children}
            </main>
        </div>
    )
}

export default Layout

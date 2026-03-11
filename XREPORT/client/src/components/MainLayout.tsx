import { NavLink, Outlet } from 'react-router-dom';
import { BrainCircuit, FileSearch, FileStack } from 'lucide-react';
import './MainLayout.css';

const navItems = [
    { path: '/dataset', icon: FileStack, label: 'Dataset' },
    { path: '/training', icon: BrainCircuit, label: 'Training' },
    { path: '/inference', icon: FileSearch, label: 'Inference' },
];

export default function MainLayout() {
    return (
        <div className="main-layout">
            <div className="main-layout-chrome">
                <header className="app-header-bar">
                    <div className="app-header-brand">
                        <img className="app-header-logo" src="/favicon.png" alt="XREPORT logo" />
                        <div className="app-header-titles">
                            <h1 className="app-header-title">XREPORT</h1>
                            <p className="app-header-subtitle">Radiological Reports Generator</p>
                        </div>
                    </div>
                </header>

                <nav className="app-nav-bar" aria-label="Primary navigation">
                    <div className="app-nav-list">
                        {navItems.map((item) => (
                            <NavLink
                                key={item.path}
                                to={item.path}
                                className={({ isActive }) => `app-nav-button${isActive ? ' active' : ''}`}
                            >
                                <item.icon size={16} />
                                <span>{item.label}</span>
                            </NavLink>
                        ))}
                    </div>
                </nav>
            </div>

            <div className="main-layout-content">
                <Outlet />
            </div>
        </div>
    );
}

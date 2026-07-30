import { NavLink, Outlet } from 'react-router-dom';
import { BrainCircuit, FileSearch, FileStack, Settings } from 'lucide-react';
import './MainLayout.css';

const developmentNavItems = [
    { path: '/dataset', icon: FileStack, label: 'Dataset' },
    { path: '/training', icon: BrainCircuit, label: 'Training' },
];

const inferenceNavItem = { path: '/inference', icon: FileSearch, label: 'Inference' };

export default function MainLayout() {
    return (
        <div className="main-layout">
            <div className="main-layout-chrome">
                <nav className="app-nav-bar" aria-label="Primary navigation">
                    <div className="app-nav-brand">
                        <img className="app-nav-logo" src="/favicon.png" alt="XREPORT logo" />
                        <div className="app-nav-titles">
                            <h1 className="app-nav-title">XREPORT</h1>
                            <p className="app-nav-subtitle">Radiological Reports Generator</p>
                        </div>
                    </div>
                    <div className="app-nav-list">
                        <NavLink
                            to={inferenceNavItem.path}
                            title={inferenceNavItem.label}
                            aria-label={inferenceNavItem.label}
                            className={({ isActive }) => `app-nav-button app-nav-button-primary${isActive ? ' active' : ''}`}
                        >
                            <inferenceNavItem.icon size={16} />
                            <span>{inferenceNavItem.label}</span>
                        </NavLink>
                        <span className="app-nav-separator" aria-hidden="true" />
                        <span className="app-nav-group-label">Model development</span>
                        {developmentNavItems.map((item) => (
                            <NavLink
                                key={item.path}
                                to={item.path}
                                title={item.label}
                                aria-label={item.label}
                                className={({ isActive }) => `app-nav-button${isActive ? ' active' : ''}`}
                            >
                                <item.icon size={16} />
                                <span>{item.label}</span>
                            </NavLink>
                        ))}
                    </div>
                    <button
                        type="button"
                        className="app-nav-button app-nav-settings"
                        title="Settings"
                        aria-label="Settings"
                        disabled
                    >
                        <Settings size={16} />
                        <span>Settings</span>
                    </button>
                </nav>
            </div>

            <div className="main-layout-content">
                <Outlet />
            </div>
        </div>
    );
}

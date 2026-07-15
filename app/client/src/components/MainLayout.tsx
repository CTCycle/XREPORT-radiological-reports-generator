import { NavLink, Outlet } from 'react-router-dom';
import { BrainCircuit, FileSearch, FileStack } from 'lucide-react';
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
                </nav>
            </div>

            <div className="main-layout-content">
                <Outlet />
            </div>
        </div>
    );
}

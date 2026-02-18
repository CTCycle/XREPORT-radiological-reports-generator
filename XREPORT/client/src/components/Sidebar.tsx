import { NavLink } from 'react-router-dom';
import { BrainCircuit, FileSearch, FileStack } from 'lucide-react';
import './Sidebar.css';

const navItems = [
    { path: '/dataset', icon: FileStack, label: 'Dataset' },
    { path: '/training', icon: BrainCircuit, label: 'Training' },
    { path: '/inference', icon: FileSearch, label: 'Inference' },
];

export default function Sidebar() {
    return (
        <div className="sidebar">
            {navItems.map((item) => (
                <NavLink
                    key={item.path}
                    to={item.path}
                    title={item.label}
                    className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}
                >
                    <item.icon size={26} />
                    <span className="sidebar-link-label">
                        {item.label}
                    </span>
                </NavLink>
            ))}
        </div>
    );
}

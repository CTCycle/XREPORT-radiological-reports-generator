import { NavLink } from 'react-router-dom';
import { BrainCircuit, FileSearch, Database } from 'lucide-react';

const navItems = [
    { path: '/training', icon: BrainCircuit, label: 'Training' },
    { path: '/inference', icon: FileSearch, label: 'Inference' },
    { path: '/database', icon: Database, label: 'Database' },
];

export default function Sidebar() {
    return (
        <div style={{
            width: '64px',
            backgroundColor: '#2b2b2b',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            paddingTop: '20px',
            borderRight: '1px solid #333'
        }}>
            {navItems.map((item) => (
                <NavLink
                    key={item.path}
                    to={item.path}
                    title={item.label}
                    style={({ isActive }) => ({
                        color: isActive ? '#646cff' : '#a1a1aa',
                        marginBottom: '24px',
                        padding: '10px',
                        borderRadius: '8px',
                        backgroundColor: isActive ? 'rgba(100, 108, 255, 0.1)' : 'transparent',
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        outline: 'none'
                    })}
                >
                    <item.icon size={24} />
                </NavLink>
            ))}
        </div>
    );
}

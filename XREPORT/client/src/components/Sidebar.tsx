import { NavLink } from 'react-router-dom';
import { BrainCircuit, FileSearch, Database, FileStack } from 'lucide-react';

const navItems = [
    { path: '/dataset', icon: FileStack, label: 'Dataset' },
    { path: '/training', icon: BrainCircuit, label: 'Training' },
    { path: '/inference', icon: FileSearch, label: 'Inference' },
    { path: '/database', icon: Database, label: 'Database' },
];

export default function Sidebar() {
    return (
        <div style={{
            width: '84px',
            backgroundColor: '#2b2b2b',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            paddingTop: '20px',
            borderRight: '1px solid #333',
            height: '100%',
            overflowY: 'auto',
            scrollbarWidth: 'none'
        }}>
            {navItems.map((item) => (
                <NavLink
                    key={item.path}
                    to={item.path}
                    title={item.label}
                    style={({ isActive }) => ({
                        color: isActive ? '#646cff' : '#a1a1aa',
                        marginBottom: '32px',
                        padding: '8px',
                        borderRadius: '8px',
                        backgroundColor: isActive ? 'rgba(100, 108, 255, 0.1)' : 'transparent',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'center',
                        alignItems: 'center',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        outline: 'none',
                        gap: '6px',
                        width: '70px',
                        textDecoration: 'none'
                    })}
                >
                    <item.icon size={26} />
                    <span style={{ 
                        fontSize: '11px', 
                        fontWeight: 500,
                        textAlign: 'center',
                        lineHeight: 1
                    }}>
                        {item.label}
                    </span>
                </NavLink>
            ))}
        </div>
    );
}

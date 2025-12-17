import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';

export default function MainLayout() {
    return (
        <div style={{ display: 'flex', width: '100vw', height: '100vh', overflow: 'hidden' }}>
            <Sidebar />
            <div style={{ flex: 1, overflow: 'auto', backgroundColor: '#1e1e1e', color: '#fff' }}>
                <Outlet />
            </div>
        </div>
    );
}

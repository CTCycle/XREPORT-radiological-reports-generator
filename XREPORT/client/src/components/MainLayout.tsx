import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import './MainLayout.css';

export default function MainLayout() {
    return (
        <div className="main-layout">
            <Sidebar />
            <div className="main-layout-content">
                <Outlet />
            </div>
        </div>
    );
}

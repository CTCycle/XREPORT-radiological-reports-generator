import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './components/MainLayout';
import TrainingPage from './pages/TrainingPage';
import InferencePage from './pages/InferencePage';
import DatabasePage from './pages/DatabasePage';

export default function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<MainLayout />}>
                    <Route index element={<Navigate to="/training" replace />} />
                    <Route path="training" element={<TrainingPage />} />
                    <Route path="inference" element={<InferencePage />} />
                    <Route path="database" element={<DatabasePage />} />
                </Route>
            </Routes>
        </BrowserRouter>
    );
}

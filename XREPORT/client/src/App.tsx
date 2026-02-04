import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './components/MainLayout';
import TrainingPage from './pages/TrainingPage';
import InferencePage from './pages/InferencePage';
import DatabasePage from './pages/DatabasePage';
import DatasetValidationPage from './pages/DatasetValidationPage';
import DatasetPage from './pages/DatasetPage';
import { AppStateProvider } from './AppStateContext';

export default function App() {
    return (
        <AppStateProvider>
            <BrowserRouter>
                <Routes>
                    <Route path="/" element={<MainLayout />}>
                        <Route index element={<Navigate to="/dataset" replace />} />
                        <Route path="training" element={<TrainingPage />} />
                        <Route path="inference" element={<InferencePage />} />
                        <Route path="database" element={<DatabasePage />} />
                        <Route path="dataset" element={<DatasetPage />} />
                        <Route path="dataset/validate/:datasetName" element={<DatasetValidationPage />} />
                    </Route>
                </Routes>
            </BrowserRouter>
        </AppStateProvider>
    );
}

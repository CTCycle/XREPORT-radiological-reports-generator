import { useState, useEffect, useCallback, useRef } from 'react';
import { Folder, HardDrive, ArrowUp, X, Check, Image } from 'lucide-react';
import { browseDirectory } from '../services/trainingService';
import { BrowseResponse, DirectoryItem } from '../types/trainingApi';
import './FolderBrowser.css';

interface FolderBrowserProps {
    readonly isOpen: boolean;
    readonly onClose: () => void;
    readonly onSelect: (path: string, imageCount: number) => void;
}

export default function FolderBrowser({ isOpen, onClose, onSelect }: FolderBrowserProps) {
    const [browseData, setBrowseData] = useState<BrowseResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const loadDirectory = useCallback(async (path: string, requestId: number) => {
        setLoading(true);
        setError(null);

        const { result, error: err } = await browseDirectory(path);

        if (requestId !== currentRequestRef.current || !isOpenRef.current) {
            return;
        }

        setLoading(false);

        if (err) {
            setError(err);
            return;
        }

        if (result) {
            setBrowseData(result);
        }
    }, []);

    const isOpenRef = useRef(isOpen);
    const currentRequestRef = useRef(0);
    isOpenRef.current = isOpen;

    useEffect(() => {
        if (isOpen) {
            currentRequestRef.current += 1;
            void loadDirectory('', currentRequestRef.current);
        }
    }, [isOpen, loadDirectory]);

    const currentPath = browseData?.current_path ?? '';
    const currentImageCount = browseData?.items.reduce((sum, item) => sum + item.image_count, 0) ?? 0;

    const handleItemClick = (item: DirectoryItem) => {
        if (item.is_dir) {
            currentRequestRef.current += 1;
            void loadDirectory(item.path, currentRequestRef.current);
        }
    };

    const handleGoUp = () => {
        if (browseData?.parent_path !== null) {
            currentRequestRef.current += 1;
            void loadDirectory(browseData?.parent_path || '', currentRequestRef.current);
        }
    };

    const handleSelect = () => {
        if (currentPath) {
            // Get total image count for current directory
            const totalImages = browseData?.items.reduce((sum, item) => sum + item.image_count, 0) || 0;
            onSelect(currentPath, totalImages);
            onClose();
        }
    };

    if (!isOpen) return null;

    return (
        <div className="folder-browser-overlay" role="presentation">
            <button
                type="button"
                className="folder-browser-backdrop"
                onClick={onClose}
                aria-label="Close folder browser"
            />
            <div className="folder-browser-modal" role="dialog" aria-modal="true" aria-label="Select image folder">
                <div className="folder-browser-header">
                    <h3>Select Image Folder</h3>
                    <button type="button" className="close-btn" onClick={onClose} aria-label="Close folder browser">
                        <X size={20} />
                    </button>
                </div>

                <div className="folder-browser-path">
                    <span className="path-label">Current:</span>
                    <span className="path-value">{currentPath || 'Select a drive'}</span>
                </div>

                <div className="folder-browser-content">
                    {loading && <div className="loading">Loading...</div>}

                    {error && <div className="error">{error}</div>}

                    {!loading && !error && browseData && (
                        <div className="folder-list">
                            {/* Go up button */}
                            {browseData.parent_path !== null && currentPath && (
                                <button type="button" className="folder-item go-up" onClick={handleGoUp}>
                                    <ArrowUp size={18} />
                                    <span>..</span>
                                </button>
                            )}

                            {/* Drives (when at root) */}
                            {!currentPath && browseData.drives.map((drive) => (
                                <button
                                    key={drive}
                                    type="button"
                                    className="folder-item drive"
                                    onClick={() => {
                                        currentRequestRef.current += 1;
                                        void loadDirectory(drive, currentRequestRef.current);
                                    }}
                                >
                                    <HardDrive size={18} />
                                    <span>{drive}</span>
                                </button>
                            ))}

                            {/* Directories */}
                            {currentPath && browseData.items.map((item) => (
                                <button
                                    key={item.path}
                                    type="button"
                                    className="folder-item"
                                    onClick={() => handleItemClick(item)}
                                >
                                    <Folder size={18} />
                                    <span className="folder-name">{item.name}</span>
                                    {item.image_count > 0 && (
                                        <span className="image-count">
                                            <Image size={12} />
                                            {item.image_count}
                                        </span>
                                    )}
                                </button>
                            ))}

                            {currentPath && browseData.items.length === 0 && (
                                <div className="empty-folder">No subfolders</div>
                            )}
                        </div>
                    )}
                </div>

                <div className="folder-browser-footer">
                    <div className="current-selection">
                        {currentPath && (
                            <span className="selection-info">
                                <Image size={14} />
                                {currentImageCount} images in subfolders
                            </span>
                        )}
                    </div>
                    <div className="action-buttons">
                        <button type="button" className="btn btn-secondary" onClick={onClose}>
                            Cancel
                        </button>
                        <button
                            type="button"
                            className="btn btn-primary"
                            onClick={handleSelect}
                            disabled={!currentPath}
                        >
                            <Check size={16} />
                            Select Folder
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

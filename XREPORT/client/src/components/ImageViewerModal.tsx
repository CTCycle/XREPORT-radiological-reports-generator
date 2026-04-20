import { useState, useEffect, useCallback, useRef } from 'react';
import { X, ChevronLeft, ChevronRight, AlertCircle, Loader } from 'lucide-react';
import {
    getDatasetImageCount,
    getDatasetImageMetadata,
    getDatasetImageContentUrl,
} from '../services/trainingService';
import { ImageMetadataResponse } from '../types/trainingApi';
import './ImageViewerModal.css';

interface ImageViewerModalProps {
    isOpen: boolean;
    datasetName: string | null;
    onClose: () => void;
}

export default function ImageViewerModal({ isOpen, datasetName, onClose }: ImageViewerModalProps) {
    const [currentIndex, setCurrentIndex] = useState<number>(1);
    const [totalImages, setTotalImages] = useState<number>(0);
    const [loadingCount, setLoadingCount] = useState<boolean>(false);
    const [loadingImage, setLoadingImage] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [metadata, setMetadata] = useState<ImageMetadataResponse | null>(null);
    const [imageError, setImageError] = useState<string | null>(null);
    const activeDatasetRef = useRef<string | null>(null);
    const isOpenRef = useRef(isOpen);

    isOpenRef.current = isOpen;
    activeDatasetRef.current = datasetName;

    const fetchCount = useCallback(async (dataset: string) => {
        setLoadingCount(true);
        const { result, error } = await getDatasetImageCount(dataset);
        if (!isOpenRef.current || activeDatasetRef.current !== dataset) {
            return;
        }
        setLoadingCount(false);

        if (error) {
            setError(error);
        } else if (result) {
            setTotalImages(result.count);
        }
    }, []);

    const fetchMetadata = useCallback(async (dataset: string, index: number) => {
        setLoadingImage(true);
        setImageError(null);

        const { result, error } = await getDatasetImageMetadata(dataset, index);
        if (!isOpenRef.current || activeDatasetRef.current !== dataset) {
            return;
        }

        if (error) {
            setError(error);
            setLoadingImage(false);
        } else if (result) {
            setMetadata(result);
            if (!result.valid_path) {
                setImageError(`Source file not found at ${result.path}`);
                setLoadingImage(false);
            }
        }
    }, []);

    useEffect(() => {
        if (!isOpen || !datasetName) {
            setLoadingCount(false);
            setLoadingImage(false);
            return;
        }
        setCurrentIndex(1);
        setTotalImages(0);
        setError(null);
        setImageError(null);
        setMetadata(null);
        void fetchCount(datasetName);
    }, [datasetName, fetchCount, isOpen]);

    useEffect(() => {
        if (isOpen && datasetName && totalImages > 0) {
            setError(null);
            setImageError(null);
            void fetchMetadata(datasetName, currentIndex);
        }
    }, [isOpen, datasetName, totalImages, currentIndex, fetchMetadata]);

    const handlePrev = useCallback(() => {
        if (currentIndex > 1) {
            setCurrentIndex(prev => prev - 1);
        }
    }, [currentIndex]);

    const handleNext = useCallback(() => {
        if (currentIndex < totalImages) {
            setCurrentIndex(prev => prev + 1);
        }
    }, [currentIndex, totalImages]);

    const handleKeyDown = useCallback((e: KeyboardEvent) => {
        if (!isOpen) return;
        if (e.key === 'ArrowLeft') handlePrev();
        if (e.key === 'ArrowRight') handleNext();
        if (e.key === 'Escape') onClose();
    }, [isOpen, onClose, handlePrev, handleNext]);

    useEffect(() => {
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [handleKeyDown]);

    if (!isOpen) return null;

    const imageUrl = (datasetName && metadata?.valid_path)
        ? getDatasetImageContentUrl(datasetName, currentIndex)
        : undefined;

    return (
        <div className="modal-backdrop" onClick={onClose}>
            <div className="viewer-modal" onClick={e => e.stopPropagation()}>
                <div className="viewer-header">
                    <div className="viewer-title">
                        <h3>Image Viewer</h3>
                        <p className="viewer-subtitle">
                            Dataset: <strong>{datasetName}</strong>
                            {totalImages > 0 && <span className="viewer-counter"> • {currentIndex} / {totalImages}</span>}
                        </p>
                    </div>
                    <button className="viewer-close" onClick={onClose} aria-label="Close viewer">
                        <X size={18} />
                    </button>
                </div>

                <div className="viewer-content">
                    {loadingCount ? (
                        <div className="viewer-loading">
                            <Loader className="spin" size={32} />
                            <p>Loading dataset info...</p>
                        </div>
                    ) : error ? (
                        <div className="viewer-error">
                            <AlertCircle size={32} />
                            <p>{error}</p>
                        </div>
                    ) : totalImages === 0 ? (
                        <div className="viewer-empty">
                            <p>No images found in this dataset.</p>
                        </div>
                    ) : (
                        <div className="viewer-main">
                            {/* Navigation Left */}
                            <button
                                className="nav-btn nav-prev"
                                onClick={handlePrev}
                                disabled={currentIndex <= 1}
                                aria-label="Previous image"
                            >
                                <ChevronLeft size={24} />
                            </button>

                            {/* Image Display */}
                            <div className="image-display-area">
                                {loadingImage && !imageError && (
                                    <div className="image-loader">
                                        <Loader className="spin" size={24} />
                                    </div>
                                )}

                                {imageError ? (
                                    <div className="image-error-display">
                                        <AlertCircle size={48} />
                                        <p>{imageError}</p>
                                    </div>
                                ) : (
                                    imageUrl && (
                                        <img
                                            src={imageUrl}
                                            alt={metadata?.image_name || "X-ray"}
                                            className="viewer-image"
                                            onLoad={() => setLoadingImage(false)}
                                            onError={() => {
                                                setLoadingImage(false);
                                                setImageError(`Failed to load image from ${metadata?.path || 'server'}`);
                                            }}
                                            style={{ display: loadingImage ? 'none' : 'block' }}
                                        />
                                    )
                                )}
                            </div>

                            {/* Navigation Right */}
                            <button
                                className="nav-btn nav-next"
                                onClick={handleNext}
                                disabled={currentIndex >= totalImages}
                                aria-label="Next image"
                            >
                                <ChevronRight size={24} />
                            </button>
                        </div>
                    )}
                </div>

                {/* Caption Footer */}
                {!error && metadata && (
                    <div className="viewer-footer">
                        <div className="caption-container">
                            <h4>Radiological Report</h4>
                            <p>{metadata.caption || "No caption available"}</p>
                            <div className="image-meta">
                                <small>Filename: {metadata.image_name}</small>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

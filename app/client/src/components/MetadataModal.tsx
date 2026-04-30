import { X } from 'lucide-react';
import { MetadataModalState } from '../types/metadata';
import './MetadataModal.css';

export interface MetadataModalProps {
    state: MetadataModalState | null;
    onClose: () => void;
}

export default function MetadataModal({ state, onClose }: MetadataModalProps) {
    if (!state) return null;

    return (
        <div className="metadata-backdrop">
            <div className="metadata-modal">
                <div className="metadata-header">
                    <div>
                        <h3>{state.title}</h3>
                        {state.subtitle && <p className="metadata-subtitle">{state.subtitle}</p>}
                    </div>
                    <button className="metadata-close" onClick={onClose} aria-label="Close metadata dialog">
                        <X size={18} />
                    </button>
                </div>
                <div className="metadata-body">
                    {state.error && <div className="metadata-error">{state.error}</div>}
                    {!state.error && state.sections?.map((section) => (
                        <div className="metadata-section" key={section.title}>
                            <h4>{section.title}</h4>
                            <div className="metadata-grid">
                                {section.entries.map((entry) => (
                                    <div className="metadata-row" key={`${section.title}-${entry.label}`}>
                                        <span className="metadata-label">{entry.label}</span>
                                        <span className="metadata-value">{entry.value}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
                <div className="metadata-footer">
                    <button className="btn btn-secondary" type="button" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
}

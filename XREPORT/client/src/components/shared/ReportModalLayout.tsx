import { ReactNode } from 'react';
import { X } from 'lucide-react';

export interface ReportModalChip {
    id: string;
    icon: ReactNode;
    text: string;
}

interface ReportModalLayoutProps {
    isOpen: boolean;
    title: string;
    subtitleLabel: string;
    subtitleValue: string;
    chips: ReportModalChip[];
    metrics: string[];
    onClose: () => void;
    children: ReactNode;
}

export default function ReportModalLayout({
    isOpen,
    title,
    subtitleLabel,
    subtitleValue,
    chips,
    metrics,
    onClose,
    children,
}: ReportModalLayoutProps) {
    if (!isOpen) return null;

    return (
        <div className="modal-backdrop" onClick={onClose}>
            <div className="report-modal" onClick={(e) => e.stopPropagation()}>
                <div className="report-header">
                    <div>
                        <h3>{title}</h3>
                        <p className="report-subtitle">
                            {subtitleLabel}: <strong>{subtitleValue}</strong>
                        </p>
                    </div>
                    <button className="report-close" onClick={onClose} aria-label={`Close ${title}`}>
                        <X size={18} />
                    </button>
                </div>

                <div className="report-meta">
                    {chips.map((chip) => (
                        <div key={chip.id} className="report-chip">
                            {chip.icon}
                            <span>{chip.text}</span>
                        </div>
                    ))}
                </div>

                <div className="report-metrics">
                    {metrics.map(metric => (
                        <span key={metric} className="report-metric-pill">
                            {metric}
                        </span>
                    ))}
                </div>

                <div className="report-body">
                    {children}
                </div>
            </div>
        </div>
    );
}

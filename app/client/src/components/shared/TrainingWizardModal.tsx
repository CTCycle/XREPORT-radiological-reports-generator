import { ReactNode } from 'react';
import { X } from 'lucide-react';

interface TrainingWizardModalProps {
    title: string;
    subtitle?: ReactNode;
    onClose: () => void;
    body: ReactNode;
    footer: ReactNode;
    steps?: ReactNode;
}

export default function TrainingWizardModal({
    title,
    subtitle,
    onClose,
    body,
    footer,
    steps,
}: TrainingWizardModalProps) {
    return (
        <div className="training-modal-backdrop">
            <div className="training-wizard-modal">
                <div className="training-wizard-header">
                    <div>
                        <h3>{title}</h3>
                        {subtitle}
                    </div>
                    <button className="training-wizard-close" onClick={onClose} aria-label="Close wizard">
                        <X size={18} />
                    </button>
                </div>
                {steps}
                <div className="training-wizard-body">{body}</div>
                <div className="training-wizard-footer">{footer}</div>
            </div>
        </div>
    );
}


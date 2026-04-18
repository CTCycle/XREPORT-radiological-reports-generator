import { MouseEventHandler, ReactNode } from 'react';

interface IconActionButtonProps {
    className: string;
    title: string;
    onClick: MouseEventHandler<HTMLButtonElement>;
    children: ReactNode;
    ariaLabel?: string;
    disabled?: boolean;
    type?: 'button' | 'submit' | 'reset';
}

export default function IconActionButton({
    className,
    title,
    onClick,
    children,
    ariaLabel,
    disabled = false,
    type = 'button',
}: IconActionButtonProps) {
    return (
        <button
            type={type}
            className={className}
            title={title}
            onClick={onClick}
            aria-label={ariaLabel}
            disabled={disabled}
        >
            {children}
        </button>
    );
}

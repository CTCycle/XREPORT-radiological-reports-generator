interface FormCheckboxProps {
    checked: boolean;
    label: string;
    onChange: (checked: boolean) => void;
    disabled?: boolean;
    className?: string;
}

export default function FormCheckbox({
    checked,
    label,
    onChange,
    disabled = false,
    className,
}: FormCheckboxProps) {
    const classes = className ? `form-checkbox ${className}` : 'form-checkbox';
    return (
        <label className={classes}>
            <input
                type="checkbox"
                checked={checked}
                disabled={disabled}
                onChange={(event) => onChange(event.target.checked)}
            />
            <div className="checkbox-visual" />
            <span className="checkbox-label">{label}</span>
        </label>
    );
}

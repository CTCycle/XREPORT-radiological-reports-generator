import { useEffect } from 'react';

export function useResetOnOpen(isOpen: boolean, onOpen: () => void): void {
    useEffect(() => {
        if (!isOpen) {
            return;
        }
        onOpen();
    }, [isOpen, onOpen]);
}


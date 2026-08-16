export interface MetadataEntry {
    label: string;
    value: string;
};

export interface MetadataSection {
    title: string;
    entries: MetadataEntry[];
};

export interface MetadataModalState {
    title: string;
    subtitle?: string;
    sections?: MetadataSection[];
    error?: string;
};

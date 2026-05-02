export type MetadataEntry = {
    label: string;
    value: string;
};

export type MetadataSection = {
    title: string;
    entries: MetadataEntry[];
};

export type MetadataModalState = {
    title: string;
    subtitle?: string;
    sections?: MetadataSection[];
    error?: string;
};

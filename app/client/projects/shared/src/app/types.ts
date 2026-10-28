import type { DraftSections } from './common/report-sections';
import type { GenerationProfile, InferenceHistoryDetail, ModelAvailability } from './types/inferenceApi';

// ============================================================================
// Inference Page State
// ============================================================================
// One model's working study. The active model's study lives inline in InferencePageState;
// the studies of the other models are kept aside so switching back restores them unchanged.
export interface InferenceStudy {
    images: File[];
    currentIndex: number;
    generatedReport: string;
    // 'selected' runs the previewed image, 'all' runs every image independently, 'study' sends all images as one study
    runScope: 'selected' | 'all' | 'study';
    // Uploaded-image indices submitted by the active or latest run, in submission order
    runIndices: number[];
    // Whether the latest run produced one study-level report shared by every submitted image
    studyReport: boolean;
    // Saved history request of the latest completed run
    requestId: string | null;
    generationProfile: GenerationProfile;
    clinicalContext: string;
    // Per-image reports for synchronized navigation
    reports: Record<number, string>;
    // Editable report sections by image index, and the saved sections they are compared against
    drafts: Record<number, DraftSections>;
    baselineDrafts: Record<number, DraftSections>;
    persistedHistory: InferenceHistoryDetail | null;
}

export interface InferencePageState extends InferenceStudy {
    isGenerating: boolean;
    // Survives page navigation so the inference page can resume tracking the running job
    activeJobId: string | null;
    isCopied: boolean;
    selectedModelRef: string;
    modelAvailability: ModelAvailability[];
    isLoadingModels: boolean;
    // Streaming state
    streamingTokens: string;
    currentStreamingIndex: number;
}


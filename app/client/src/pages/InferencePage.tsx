import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from 'react';
import {
    AlertTriangle, Check, ChevronLeft, ChevronRight, Copy, Download,
    FileImage, ImagePlus, Loader2, RefreshCw, Search, Sparkles, Trash2,
} from 'lucide-react';
import './InferencePage.css';
import { useInferencePageState } from '../AppStateContext';
import type { GenerationProfile, ModelAvailability } from '../types/inferenceApi';
import { useAsyncJob } from '../hooks/useAsyncJob';
import { asRecord, readString, readStringArray } from '../common/parsers';
import { generateReports, getInferenceJobStatus, getInferenceModels } from '../services/inferenceService';

type DraftSections = { findings: string; impression: string };
type GenerationRequest = {
    images: File[];
    modelRef: string;
    generationProfile: GenerationProfile;
    clinicalContext: string;
};

const EMPTY_DRAFT: DraftSections = { findings: '', impression: '' };

function readStringMap(value: unknown): Record<string, string> | undefined {
    const record = asRecord(value);
    if (!record) return undefined;
    const entries = Object.entries(record);
    if (entries.some(([, entry]) => readString(entry) === undefined)) return undefined;
    return Object.fromEntries(entries.map(([key, entry]) => [key, readString(entry) ?? '']));
}

function toReportsByIndex(result: unknown, images: File[]): Record<number, string> {
    const payload = asRecord(result);
    if (!payload) return {};
    const reports = readStringMap(payload.reports);
    const ordered = readStringArray(payload.reports_ordered);
    const filenames = readStringArray(payload.report_filenames);
    if (ordered?.length) return Object.fromEntries(ordered.map((report, index) => [index, report]));
    if (!reports) return {};
    const names = filenames?.length ? filenames : images.map(image => image.name);
    const mapped = Object.fromEntries(names.flatMap((name, index) => reports[name] === undefined ? [] : [[index, reports[name]]]));
    return Object.keys(mapped).length ? mapped : Object.fromEntries(Object.values(reports).map((report, index) => [index, report]));
}

function parseDraft(report: string): DraftSections {
    const normalized = report.trim();
    if (!normalized) return EMPTY_DRAFT;
    const findingsMatch = normalized.match(/(?:^|\n)\s*(?:#{1,3}\s*)?findings\s*:?\s*([\s\S]*?)(?=\n\s*(?:#{1,3}\s*)?impression\s*:?|$)/i);
    const impressionMatch = normalized.match(/(?:^|\n)\s*(?:#{1,3}\s*)?impression\s*:?\s*([\s\S]*)$/i);
    if (!findingsMatch && !impressionMatch) return { findings: normalized, impression: '' };
    return {
        findings: findingsMatch?.[1]?.trim() ?? '',
        impression: impressionMatch?.[1]?.trim() ?? '',
    };
}

function formatDraft(draft: DraftSections): string {
    return `Findings\n${draft.findings.trim()}\n\nImpression\n${draft.impression.trim()}`.trim();
}

function parseProfile(value: string): GenerationProfile {
    return value === 'concise' || value === 'detailed' ? value : 'deterministic';
}

export default function InferencePage() {
    const {
        state, setImages, setCurrentIndex, setGeneratedReport, setIsGenerating,
        setIsCopied, clearImages, setSelectedModelRef, setGenerationProfile,
        setClinicalContext, setModelAvailability, setIsLoadingModels, setReports,
        setStreamingTokens, setCurrentStreamingIndex,
    } = useInferencePageState();
    const [modelFilter, setModelFilter] = useState('');
    const [providerFilter, setProviderFilter] = useState('all');
    const [catalogError, setCatalogError] = useState<string | null>(null);
    const [drafts, setDrafts] = useState<Record<number, DraftSections>>({});
    const fileInputRef = useRef<HTMLInputElement>(null);

    const selectedModel = useMemo(
        () => state.modelAvailability.find(model => model.model_ref === state.selectedModelRef) ?? null,
        [state.modelAvailability, state.selectedModelRef],
    );
    const maxImages = selectedModel?.input_semantics === 'independent_images' ? 16 : 1;
    const activeDraft = drafts[state.currentIndex] ?? parseDraft(state.generatedReport);
    const filteredModels = useMemo(() => {
        const query = modelFilter.trim().toLowerCase();
        return state.modelAvailability.filter(model =>
            (providerFilter === 'all' || model.provider === providerFilter)
            && (!query || `${model.display_name} ${model.description} ${model.provider}`.toLowerCase().includes(query))
        );
    }, [modelFilter, providerFilter, state.modelAvailability]);

    const generationJob = useAsyncJob({
        startJob: (request: GenerationRequest) => generateReports(
            request.images, request.modelRef, request.generationProfile, request.clinicalContext,
        ),
        getStatus: getInferenceJobStatus,
        onUpdate: status => {
            const reports = toReportsByIndex(status.result, state.images);
            if (!Object.keys(reports).length) return;
            setReports(reports);
            setDrafts(Object.fromEntries(Object.entries(reports).map(([index, report]) => [Number(index), parseDraft(report)])));
            setGeneratedReport(reports[state.currentIndex] ?? '');
        },
        onComplete: () => setIsGenerating(false),
    });

    useEffect(() => {
        const loadModels = async () => {
            setIsLoadingModels(true);
            setCatalogError(null);
            const { result, error } = await getInferenceModels();
            if (result) {
                setModelAvailability(result.models);
                const selectedReady = result.models.some(model => model.model_ref === state.selectedModelRef && model.status === 'ready');
                if (!selectedReady) setSelectedModelRef(result.models.find(model => model.status === 'ready')?.model_ref ?? '');
            } else {
                setCatalogError(error ?? 'Unable to load the local model catalog.');
            }
            setIsLoadingModels(false);
        };
        void loadModels();
    }, []);

    useEffect(() => {
        const report = state.reports[state.currentIndex] ?? '';
        setGeneratedReport(report);
    }, [state.currentIndex, state.reports, setGeneratedReport]);

    const resetStudy = () => {
        clearImages();
        setDrafts({});
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const addFiles = (files: FileList | null) => {
        if (!files?.length) return;
        const accepted = Array.from(files).filter(file => file.type.startsWith('image/'));
        const next = maxImages === 1 ? accepted.slice(0, 1) : [...state.images, ...accepted].slice(0, maxImages);
        setImages(next);
        setCurrentIndex(0);
        setReports({});
        setDrafts({});
        setGeneratedReport('');
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const onDrop = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        addFiles(event.dataTransfer.files);
    };

    const generate = async () => {
        if (!selectedModel || selectedModel.status !== 'ready' || !state.images.length) return;
        setIsGenerating(true);
        setReports({});
        setDrafts({});
        setGeneratedReport('');
        setStreamingTokens('');
        setCurrentStreamingIndex(-1);
        const started = await generationJob.start({
            images: state.images,
            modelRef: selectedModel.model_ref,
            generationProfile: state.generationProfile,
            clinicalContext: state.clinicalContext,
        });
        if (!started) setIsGenerating(false);
    };

    const updateDraft = (field: keyof DraftSections, value: string) => {
        const next = { ...activeDraft, [field]: value };
        setDrafts(previous => ({ ...previous, [state.currentIndex]: next }));
        setGeneratedReport(formatDraft(next));
    };

    const copyDraft = async () => {
        await navigator.clipboard.writeText(formatDraft(activeDraft));
        setIsCopied(true);
        globalThis.setTimeout(() => setIsCopied(false), 1600);
    };

    const exportDraft = () => {
        const metadata = [
            'XREPORT — RESEARCH USE ONLY — NOT CLINICALLY APPROVED',
            `Model: ${selectedModel?.display_name ?? 'Unknown'} (${selectedModel?.model_ref ?? 'Unknown'})`,
            `Revision: ${selectedModel?.model_revision ?? 'Not reported'}`,
            `Generation profile: ${state.generationProfile}`,
            `Image: ${state.images[state.currentIndex]?.name ?? 'Unknown'}`,
            '', formatDraft(activeDraft),
        ].join('\n');
        const url = URL.createObjectURL(new Blob([metadata], { type: 'text/plain;charset=utf-8' }));
        const link = document.createElement('a');
        link.href = url;
        link.download = `xreport-draft-${state.currentIndex + 1}.txt`;
        link.click();
        URL.revokeObjectURL(url);
    };

    const currentImage = state.images[state.currentIndex] ?? null;
    const currentImageUrl = useMemo(() => currentImage ? URL.createObjectURL(currentImage) : null, [currentImage]);
    useEffect(() => () => { if (currentImageUrl) URL.revokeObjectURL(currentImageUrl); }, [currentImageUrl]);

    return (
        <main className="inference-workspace">
            <header className="workspace-heading">
                <div><span className="eyebrow">Local inference</span><h1>Report drafting workspace</h1><p>Prepare a study, choose a local model, and review an editable radiology draft.</p></div>
                <div className="research-warning" role="alert"><AlertTriangle aria-hidden="true" /><div><strong>Research use only</strong><span>Models and generated drafts are not clinically approved. Qualified review and independent verification are required.</span></div></div>
            </header>

            <section className="workspace-grid">
                <aside className="catalog-panel" aria-label="Model catalog">
                    <div className="section-heading"><div><span className="step-number">1</span><h2>Select model</h2></div><span className="catalog-count">{filteredModels.length}</span></div>
                    <label className="search-field"><Search aria-hidden="true" /><span className="sr-only">Filter models</span><input value={modelFilter} onChange={event => setModelFilter(event.target.value)} placeholder="Filter models" /></label>
                    <div className="provider-tabs" aria-label="Provider filter">
                        {['all', 'xreport', 'ollama', 'huggingface'].map(provider => <button key={provider} type="button" className={providerFilter === provider ? 'active' : ''} onClick={() => setProviderFilter(provider)}>{provider}</button>)}
                    </div>
                    {state.isLoadingModels && <div className="catalog-state"><Loader2 className="spin" />Discovering local models…</div>}
                    {catalogError && <div className="catalog-state error">{catalogError}</div>}
                    <div className="model-list">
                        {filteredModels.map(model => (
                            <button type="button" key={model.model_ref} className={`model-card ${state.selectedModelRef === model.model_ref ? 'selected' : ''}`} onClick={() => setSelectedModelRef(model.model_ref)} aria-pressed={state.selectedModelRef === model.model_ref}>
                                <span className={`status-dot ${model.status}`} aria-hidden="true" /><span className="model-card-body"><strong>{model.display_name}</strong><small>{model.provider} · {model.parameter_size ?? model.category}</small></span><span className={`status-label ${model.status}`}>{model.status.replace('_', ' ')}</span>
                            </button>
                        ))}
                        {!state.isLoadingModels && !filteredModels.length && <div className="catalog-state">No models match this filter.</div>}
                    </div>
                    {selectedModel && <ModelDetails model={selectedModel} />}
                </aside>

                <section className="study-panel" aria-label="Study preparation">
                    <div className="section-heading"><div><span className="step-number">2</span><h2>Prepare study</h2></div>{state.images.length > 0 && <button type="button" className="text-button danger" onClick={resetStudy}><Trash2 />Clear</button>}</div>
                    <div className="upload-zone" onDragOver={event => event.preventDefault()} onDrop={onDrop}>
                        {currentImageUrl ? <div className="image-stage"><img src={currentImageUrl} alt={`Study image ${state.currentIndex + 1}`} /><span>{currentImage?.name}</span></div> : <button type="button" className="upload-prompt" onClick={() => fileInputRef.current?.click()}><ImagePlus /><strong>Add study image</strong><span>Drop an image here or browse local files</span></button>}
                        <input ref={fileInputRef} className="sr-only" type="file" accept="image/*" multiple={maxImages > 1} onChange={(event: ChangeEvent<HTMLInputElement>) => addFiles(event.target.files)} />
                    </div>
                    <div className="study-toolbar">
                        <button type="button" className="secondary-button" onClick={() => fileInputRef.current?.click()} disabled={!selectedModel}><ImagePlus />{state.images.length ? 'Replace / add' : 'Browse images'}</button>
                        <span>{selectedModel?.input_semantics === 'independent_images' ? `Up to ${maxImages} independent images` : 'Single-image model'}</span>
                    </div>
                    {state.images.length > 1 && <div className="image-navigation"><button type="button" aria-label="Previous image" onClick={() => setCurrentIndex(Math.max(0, state.currentIndex - 1))} disabled={state.currentIndex === 0}><ChevronLeft /></button><span>{state.currentIndex + 1} / {state.images.length}</span><button type="button" aria-label="Next image" onClick={() => setCurrentIndex(Math.min(state.images.length - 1, state.currentIndex + 1))} disabled={state.currentIndex === state.images.length - 1}><ChevronRight /></button></div>}
                    <label className="field-label" htmlFor="clinical-context"><span>Clinical context</span><small>{selectedModel?.capabilities.clinical_context ? 'Optional context supported' : 'Not supported by selected model'}</small></label>
                    <textarea id="clinical-context" className="context-input" value={state.clinicalContext} onChange={event => setClinicalContext(event.target.value)} disabled={!selectedModel?.capabilities.clinical_context || state.isGenerating} placeholder="Indication, relevant history, comparison details…" />
                    <div className="generation-controls"><label htmlFor="profile-select">Generation profile</label><select id="profile-select" value={state.generationProfile} onChange={event => setGenerationProfile(parseProfile(event.target.value))} disabled={state.isGenerating}><option value="deterministic">Deterministic</option><option value="concise">Concise</option><option value="detailed">Detailed</option></select></div>
                    <button type="button" className="generate-button" onClick={() => void generate()} disabled={!selectedModel || selectedModel.status !== 'ready' || !state.images.length || state.isGenerating}>{state.isGenerating ? <><Loader2 className="spin" />Generating draft…</> : <><Sparkles />Generate draft</>}</button>
                    {generationJob.error && <div className="generation-error" role="alert">{generationJob.error}</div>}
                </section>

                <section className="draft-panel" aria-label="Report draft">
                    <div className="section-heading"><div><span className="step-number">3</span><h2>Review draft</h2></div><div className="draft-actions"><button type="button" aria-label="Regenerate draft" title="Regenerate" onClick={() => void generate()} disabled={!state.images.length || state.isGenerating}><RefreshCw /></button><button type="button" aria-label="Copy draft" title="Copy" onClick={() => void copyDraft()} disabled={!activeDraft.findings && !activeDraft.impression}>{state.isCopied ? <Check /> : <Copy />}</button><button type="button" aria-label="Export draft" title="Export text" onClick={exportDraft} disabled={!activeDraft.findings && !activeDraft.impression}><Download /></button></div></div>
                    {!activeDraft.findings && !activeDraft.impression && !state.isGenerating ? <div className="draft-empty"><FileImage /><strong>No draft yet</strong><span>Choose a ready model and add an image to begin.</span></div> : <div className="draft-editor"><label htmlFor="findings">Findings</label><textarea id="findings" value={activeDraft.findings} onChange={event => updateDraft('findings', event.target.value)} placeholder="Generated findings will appear here." /><label htmlFor="impression">Impression</label><textarea id="impression" value={activeDraft.impression} onChange={event => updateDraft('impression', event.target.value)} placeholder="Generated impression will appear here." /></div>}
                    <div className="runtime-metadata"><span><strong>Model</strong>{selectedModel?.display_name ?? 'Not selected'}</span><span><strong>Provider</strong>{selectedModel?.provider ?? '—'}</span><span><strong>Revision</strong>{selectedModel?.model_revision?.slice(0, 12) ?? 'Not reported'}</span><span><strong>Profile</strong>{state.generationProfile}</span></div>
                </section>
            </section>
        </main>
    );
}

function ModelDetails({ model }: Readonly<{ model: ModelAvailability }>) {
    const capabilities = [
        model.capabilities.clinical_context && 'Clinical context',
        model.capabilities.multiple_current_views && 'Multiple views',
        model.capabilities.findings && 'Findings',
        model.capabilities.impression && 'Impression',
        model.capabilities.grounding && 'Grounding',
    ].filter(Boolean);
    return <div className="model-details"><div><span className="provider-pill">{model.provider}</span>{model.recommended && <span className="recommended-pill">Recommended</span>}</div><h3>{model.display_name}</h3><p>{model.description}</p><dl><div><dt>Input</dt><dd>{model.input_semantics.replace(/_/g, ' ')}</dd></div><div><dt>Revision</dt><dd>{model.model_revision?.slice(0, 12) ?? 'Not configured'}</dd></div></dl><div className="capability-list">{capabilities.map(capability => <span key={String(capability)}>{capability}</span>)}</div></div>;
}

import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from 'react';
import {
    AlertTriangle, Check, ChevronLeft, ChevronRight, Copy, Download,
    FileImage, ImagePlus, Loader2, RefreshCw, Search, Sparkles, Trash2,
} from 'lucide-react';
import './InferencePage.css';
import { useInferencePageState } from '../AppStateContext';
import type { GenerationProfile, ModelAvailability, OutputSection } from '../types/inferenceApi';
import { useAsyncJob } from '../hooks/useAsyncJob';
import { asRecord, readString, readStringArray } from '../common/parsers';
import { cancelInferenceJob, generateReports, getInferenceJobStatus, getInferenceModels } from '../services/inferenceService';

type DraftSections = Partial<Record<OutputSection, string>>;
type GenerationRequest = {
    images: File[];
    modelRef: string;
    generationProfile: GenerationProfile;
    clinicalContext: string;
};

const EMPTY_DRAFT: DraftSections = {};

const SECTION_LABELS: Record<OutputSection, string> = {
    raw_report: 'Raw report',
    findings: 'Findings',
    impression: 'Impression',
};

function readStringMap(value: unknown): Record<string, string> | undefined {
    const record = asRecord(value);
    if (!record) return undefined;
    const entries = Object.entries(record);
    if (entries.some(([, entry]) => readString(entry) === undefined)) return undefined;
    return Object.fromEntries(entries.map(([key, entry]) => [key, readString(entry) ?? '']));
}

function readSectionMap(value: unknown, images: File[]): Record<number, DraftSections> {
    const payload = asRecord(value);
    if (!payload) return {};
    const filenames = readStringArray(payload.report_filenames) ?? images.map(image => image.name);
    const byFilename = asRecord(payload.display_sections);
    if (!byFilename) return {};
    return Object.fromEntries(filenames.flatMap((filename, index) => {
        const sections = asRecord(byFilename[filename]);
        if (!sections) return [];
        const normalized = Object.fromEntries(Object.entries(sections).flatMap(([key, raw]) => {
            const text = readString(raw);
            return text === undefined ? [] : [[key, text]];
        })) as DraftSections;
        return Object.keys(normalized).length ? [[index, normalized]] : [];
    }));
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

function parseDeclaredDraft(report: string, outputSections: OutputSection[]): DraftSections {
    const normalized = report.trim();
    if (!normalized) return EMPTY_DRAFT;
    if (outputSections.includes('raw_report')) return { raw_report: report };
    const findingsMatch = normalized.match(/(?:^|\n)\s*(?:#{1,3}\s*)?findings\s*:?\s*([\s\S]*?)(?=\n\s*(?:#{1,3}\s*)?impression\s*:?|$)/i);
    const impressionMatch = normalized.match(/(?:^|\n)\s*(?:#{1,3}\s*)?impression\s*:?\s*([\s\S]*)$/i);
    const draft: DraftSections = {};
    if (outputSections.includes('findings')) draft.findings = findingsMatch?.[1]?.trim() ?? (outputSections.length === 1 ? normalized : '');
    if (outputSections.includes('impression')) draft.impression = impressionMatch?.[1]?.trim() ?? (outputSections.length === 1 ? normalized : '');
    return draft;
}

function formatDeclaredDraft(draft: DraftSections, outputSections: OutputSection[]): string {
        if (outputSections.includes('raw_report')) return draft.raw_report ?? '';
    return outputSections
        .map(section => `${SECTION_LABELS[section]}\n${draft[section]?.trim() ?? ''}`)
        .join('\n\n')
        .trim();
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
    const [generationProvenance, setGenerationProvenance] = useState<Record<string, string>>({});
    const fileInputRef = useRef<HTMLInputElement>(null);

    const selectedModel = useMemo(
        () => state.modelAvailability.find(model => model.model_ref === state.selectedModelRef) ?? null,
        [state.modelAvailability, state.selectedModelRef],
    );
    const maxImages = selectedModel?.max_current_images ?? 1;
    const outputSections = selectedModel?.output_sections ?? [];
    const activeDraft = drafts[state.currentIndex] ?? parseDeclaredDraft(state.generatedReport, outputSections);
    const hasActiveDraft = Object.values(activeDraft).some(value => Boolean(value?.trim()));
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
        cancelJob: cancelInferenceJob,
        onUpdate: status => {
            const reports = toReportsByIndex(status.result, state.images);
            if (!Object.keys(reports).length) return;
            setReports(reports);
            const provenance = asRecord(status.result?.provenance);
            if (provenance) {
                setGenerationProvenance(Object.fromEntries(
                    ['provider', 'model_ref', 'model_revision', 'adapter', 'generation_profile']
                        .flatMap(key => {
                            const value = readString(provenance[key]);
                            return value === undefined ? [] : [[key, value]];
                        }),
                ));
            }
            const sectionDrafts = readSectionMap(status.result, state.images);
            setDrafts(Object.fromEntries(Object.entries(reports).map(([index, report]) => [
                Number(index), sectionDrafts[Number(index)] ?? parseDeclaredDraft(report, outputSections),
            ])));
            setGeneratedReport(reports[state.currentIndex] ?? '');
        },
        onComplete: () => setIsGenerating(false),
        onError: () => setIsGenerating(false),
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

    const selectModel = (model: ModelAvailability) => {
        setSelectedModelRef(model.model_ref);
        if (state.images.length > model.max_current_images) {
            setImages(state.images.slice(0, model.max_current_images));
            setCurrentIndex(0);
        }
        if (!model.capabilities.clinical_context && state.clinicalContext) {
            setClinicalContext('');
        }
        setReports({});
        setDrafts({});
        setGeneratedReport('');
        setGenerationProvenance({});
    };

    const resetStudy = () => {
        clearImages();
        setDrafts({});
        setReports({});
        setGeneratedReport('');
        setGenerationProvenance({});
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const addFiles = (files: FileList | null) => {
        if (state.isGenerating || !files?.length) return;
        const accepted = Array.from(files).filter(file => file.type.startsWith('image/'));
        const next = maxImages === 1 ? accepted.slice(0, 1) : [...state.images, ...accepted].slice(0, Math.min(maxImages, 16));
        setImages(next);
        setCurrentIndex(0);
        setReports({});
        setDrafts({});
        setGeneratedReport('');
        setGenerationProvenance({});
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const onDrop = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        addFiles(event.dataTransfer.files);
    };

    const changeProfile = (profile: GenerationProfile) => {
        if (hasActiveDraft) {
            setReports({});
            setDrafts({});
            setGeneratedReport('');
            setGenerationProvenance({});
        }
        setGenerationProfile(profile);
    };

    const generate = async () => {
        if (!selectedModel || selectedModel.status !== 'ready' || !state.images.length) return;
        setIsGenerating(true);
        setReports({});
        setDrafts({});
        setGeneratedReport('');
        setGenerationProvenance({});
        setStreamingTokens('');
        setCurrentStreamingIndex(-1);
        const started = await generationJob.start({
            images: state.images,
            modelRef: selectedModel.model_ref,
            generationProfile: state.generationProfile,
            clinicalContext: selectedModel.capabilities.clinical_context ? state.clinicalContext : '',
        });
        if (!started) setIsGenerating(false);
    };

    const updateDraft = (field: OutputSection, value: string) => {
        const next = { ...activeDraft, [field]: value };
        setDrafts(previous => ({ ...previous, [state.currentIndex]: next }));
        setGeneratedReport(formatDeclaredDraft(next, outputSections));
    };

    const copyDraft = async () => {
        await navigator.clipboard.writeText(formatDeclaredDraft(activeDraft, outputSections));
        setIsCopied(true);
        globalThis.setTimeout(() => setIsCopied(false), 1600);
    };

    const exportDraft = () => {
        const metadata = [
            'XREPORT — RESEARCH USE ONLY — NOT CLINICALLY APPROVED',
            `Model: ${generationProvenance.model_ref ?? selectedModel?.display_name ?? 'Unknown'} (${generationProvenance.model_ref ?? selectedModel?.model_ref ?? 'Unknown'})`,
            `Provider: ${generationProvenance.provider ?? selectedModel?.provider ?? 'Not reported'}`,
            `Revision: ${generationProvenance.model_revision ?? selectedModel?.model_revision ?? 'Not reported'}`,
            `Generation profile: ${generationProvenance.generation_profile ?? state.generationProfile}`,
            `Image: ${state.images[state.currentIndex]?.name ?? 'Unknown'}`,
            `Adapter: ${selectedModel?.adapter ?? 'Not reported'}`,
            '', formatDeclaredDraft(activeDraft, outputSections),
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
                <div><h1>Report drafting workspace</h1><p>Prepare a study, choose a local model, and review an editable radiology draft.</p></div>
                <div className="research-warning" role="alert"><AlertTriangle aria-hidden="true" /><div><strong>Research use only</strong><span>Models and generated drafts are not clinically approved. Qualified review and independent verification are required.</span></div></div>
            </header>

            <section className="workspace-grid">
                <aside className="catalog-panel" aria-label="Model catalog">
                    <div className="section-heading"><div><span className="step-number">1</span><h2>Select model</h2></div><span className="catalog-count">{filteredModels.length}</span></div>
                    <label className="search-field"><Search aria-hidden="true" /><span className="sr-only">Filter models</span><input value={modelFilter} onChange={event => setModelFilter(event.target.value)} placeholder="Filter models" /></label>
                    <div className="provider-tabs" aria-label="Provider filter">
                        {['all', 'xreport', 'huggingface'].map(provider => <button key={provider} type="button" className={providerFilter === provider ? 'active' : ''} onClick={() => setProviderFilter(provider)}>{provider}</button>)}
                    </div>
                    {state.isLoadingModels && <div className="catalog-state"><Loader2 className="spin" />Discovering local models…</div>}
                    {catalogError && <div className="catalog-state error">{catalogError}</div>}
                    <div className="model-list">
                        {filteredModels.map(model => (
                            <button type="button" key={model.model_ref} className={`model-card ${state.selectedModelRef === model.model_ref ? 'selected' : ''}`} onClick={() => selectModel(model)} aria-pressed={state.selectedModelRef === model.model_ref} disabled={state.isGenerating}>
                                <span className={`status-dot ${model.status}`} aria-hidden="true" /><span className="model-card-body"><strong>{model.display_name}</strong><small>{model.provider} · {model.parameter_size ?? model.category}</small></span><span className={`status-label ${model.status}`}>{model.status.replace('_', ' ')}</span>
                            </button>
                        ))}
                        {!state.isLoadingModels && !filteredModels.length && <div className="catalog-state">No models match this filter.</div>}
                    </div>
                    {selectedModel && <ModelDetails model={selectedModel} />}
                </aside>

                <section className="study-panel" aria-label="Study preparation">
                    <div className="section-heading"><div><span className="step-number">2</span><h2>Prepare study</h2></div>{state.images.length > 0 && <button type="button" className="text-button danger" onClick={resetStudy} disabled={state.isGenerating}><Trash2 />Clear</button>}</div>
                    <div className="upload-zone" onDragOver={event => event.preventDefault()} onDrop={onDrop}>
                        {currentImageUrl ? <div className="image-stage"><img src={currentImageUrl} alt={`Study image ${state.currentIndex + 1}`} /><span>{currentImage?.name}</span></div> : <button type="button" className="upload-prompt" onClick={() => fileInputRef.current?.click()} disabled={state.isGenerating || !selectedModel}><ImagePlus /><strong>Add study image</strong><span>Drop an image here or browse local files</span></button>}
                        <input ref={fileInputRef} className="sr-only" type="file" accept="image/*" multiple={maxImages > 1} onChange={(event: ChangeEvent<HTMLInputElement>) => addFiles(event.target.files)} />
                    </div>
                    <div className="study-toolbar">
                        <button type="button" className="secondary-button" onClick={() => fileInputRef.current?.click()} disabled={!selectedModel || state.isGenerating}><ImagePlus />{state.images.length ? 'Replace / add' : 'Browse images'}</button>
                        <span>{selectedModel?.input_semantics === 'independent_images' ? `Up to ${maxImages} independent images` : `Up to ${maxImages} current image${maxImages === 1 ? '' : 's'}`}</span>
                    </div>
                    {state.images.length > 1 && <div className="image-navigation"><button type="button" aria-label="Previous image" onClick={() => setCurrentIndex(Math.max(0, state.currentIndex - 1))} disabled={state.currentIndex === 0}><ChevronLeft /></button><span>{state.currentIndex + 1} / {state.images.length}</span><button type="button" aria-label="Next image" onClick={() => setCurrentIndex(Math.min(state.images.length - 1, state.currentIndex + 1))} disabled={state.currentIndex === state.images.length - 1}><ChevronRight /></button></div>}
                    <label className="field-label" htmlFor="clinical-context"><span>Clinical context</span><small>{selectedModel?.capabilities.clinical_context ? 'Optional context supported' : 'Not supported by selected model'}</small></label>
                    <textarea id="clinical-context" className="context-input" value={state.clinicalContext} onChange={event => setClinicalContext(event.target.value)} disabled={!selectedModel?.capabilities.clinical_context || state.isGenerating} placeholder="Indication, relevant history, comparison details…" />
                    <div className="generation-controls"><label htmlFor="profile-select">Generation profile</label><select id="profile-select" value={state.generationProfile} onChange={event => changeProfile(parseProfile(event.target.value))} disabled={state.isGenerating || hasActiveDraft}><option value="deterministic">Deterministic</option><option value="concise">Concise</option><option value="detailed">Detailed</option></select></div>
                    <button type="button" className="generate-button" onClick={() => void generate()} disabled={!selectedModel || selectedModel.status !== 'ready' || !state.images.length || state.isGenerating}>{state.isGenerating ? <><Loader2 className="spin" />Generating draft…</> : <><Sparkles />Generate draft</>}</button>
                    {state.isGenerating && <button type="button" className="secondary-button" onClick={() => void generationJob.cancel()} disabled={!generationJob.jobId}>Cancel generation</button>}
                    {generationJob.error && <div className="generation-error" role="alert">{generationJob.error}</div>}
                </section>

                <section className="draft-panel" aria-label="Report draft">
                    <div className="section-heading"><div><span className="step-number">3</span><h2>Review draft</h2></div><div className="draft-actions"><button type="button" aria-label="Regenerate draft" title="Regenerate" onClick={() => void generate()} disabled={!selectedModel || selectedModel.status !== 'ready' || !state.images.length || state.isGenerating}><RefreshCw /></button><button type="button" aria-label="Copy draft" title="Copy" onClick={() => void copyDraft()} disabled={!hasActiveDraft}>{state.isCopied ? <Check /> : <Copy />}</button><button type="button" aria-label="Export draft" title="Export text" onClick={exportDraft} disabled={!hasActiveDraft}><Download /></button></div></div>
                    {!hasActiveDraft && !state.isGenerating ? <div className="draft-empty"><FileImage /><strong>No draft yet</strong><span>Choose a ready model and add an image to begin.</span></div> : <div className="draft-editor">
                        {outputSections.map(section => <div key={section} className="declared-section"><label htmlFor={`report-${section}`}>{SECTION_LABELS[section]}</label><textarea id={`report-${section}`} value={activeDraft[section] ?? ''} onChange={event => updateDraft(section, event.target.value)} placeholder={`${SECTION_LABELS[section]} will appear here.`} /></div>)}
                    </div>}
                    <div className="runtime-metadata"><span><strong>Model</strong>{generationProvenance.model_ref ?? selectedModel?.display_name ?? 'Not selected'}</span><span><strong>Provider</strong>{generationProvenance.provider ?? selectedModel?.provider ?? '—'}</span><span><strong>Revision</strong>{generationProvenance.model_revision ?? selectedModel?.model_revision ?? 'Not reported'}</span><span><strong>Adapter</strong>{generationProvenance.adapter ?? selectedModel?.adapter ?? 'Not reported'}</span><span><strong>Profile</strong>{generationProvenance.generation_profile ?? state.generationProfile}</span><span><strong>Output</strong>{outputSections.map(section => SECTION_LABELS[section]).join(', ') || 'Not declared'}</span></div>
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
    return <div className="model-details"><div><span className="provider-pill">{model.provider}</span>{model.recommended && <span className="recommended-pill">Recommended</span>}</div><h3>{model.display_name}</h3><p>{model.description}</p>{model.status_message && <p className="catalog-state">{model.status_message}</p>}<dl><div><dt>Status</dt><dd>{model.status.replace('_', ' ')}</dd></div><div><dt>Validation</dt><dd>{model.validation_status}</dd></div><div><dt>Input</dt><dd>{model.input_semantics.replace(/_/g, ' ')}</dd></div><div><dt>Revision</dt><dd>{model.model_revision ?? 'Not configured'}</dd></div><div><dt>Adapter</dt><dd>{model.adapter ?? 'Not reported'}</dd></div><div><dt>Loader</dt><dd>{model.model_loader ?? 'Not reported'} / {model.processor_loader ?? 'Not reported'}</dd></div><div><dt>Output</dt><dd>{model.output_sections.map(section => SECTION_LABELS[section]).join(', ') || 'Not declared'}</dd></div>{model.license && <div><dt>Licence</dt><dd>{model.license}</dd></div>}</dl><div className="capability-list">{capabilities.map(capability => <span key={String(capability)}>{capability}</span>)}</div></div>;
}

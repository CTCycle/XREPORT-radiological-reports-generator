import { useCallback, useEffect, useMemo, useState } from 'react';
import { RefreshCcw, Database as DatabaseIcon } from 'lucide-react';
import './DatabaseBrowserPage.css';
import { fetchBrowseConfig, fetchTableData, fetchTableList, TableInfo } from '../services/databaseBrowser';

interface BrowserState {
    tables: TableInfo[];
    selectedTable: string;
    rows: Record<string, unknown>[];
    columns: string[];
    rowCount: number;
    columnCount: number;
    displayName: string;
    limit: number;
    offset: number;
    loading: boolean;
    error: string | null;
    tablesLoaded: boolean;
    dataLoaded: boolean;  // Track if initial data has been loaded
}

const DEFAULT_BATCH_SIZE = 200;

export default function DatabaseBrowserPage() {
    const [state, setState] = useState<BrowserState>({
        tables: [],
        selectedTable: '',
        rows: [],
        columns: [],
        rowCount: 0,
        columnCount: 0,
        displayName: '',
        limit: DEFAULT_BATCH_SIZE,
        offset: 0,
        loading: false,
        error: null,
        tablesLoaded: false,
        dataLoaded: false,
    });

    const pageIndex = useMemo(() => Math.floor(state.offset / state.limit), [state.offset, state.limit]);
    const pageStart = useMemo(() => (state.rowCount === 0 ? 0 : state.offset + 1), [state.offset, state.rowCount]);
    const pageEnd = useMemo(
        () => Math.min(state.offset + state.rows.length, state.rowCount),
        [state.offset, state.rows.length, state.rowCount],
    );

    const canGoPrevious = useMemo(() => state.offset > 0 && !state.loading, [state.offset, state.loading]);
    const canGoNext = useMemo(
        () => state.offset + state.limit < state.rowCount && !state.loading,
        [state.offset, state.limit, state.rowCount, state.loading],
    );

    const loadTableData = useCallback(
        async (tableName: string, limit: number, offset: number) => {
            if (!tableName) return;
            setState(prev => ({ ...prev, selectedTable: tableName, limit, offset, loading: true, error: null }));
            const { result, error } = await fetchTableData(tableName, limit, offset);
            if (error || !result) {
                setState(prev => ({
                    ...prev,
                    selectedTable: tableName,
                    rows: [],
                    columns: [],
                    rowCount: 0,
                    columnCount: 0,
                    displayName: '',
                    loading: false,
                    dataLoaded: true,
                    error: error ?? 'Unable to load table.',
                }));
                return;
            }
            setState(prev => ({
                ...prev,
                selectedTable: tableName,
                rows: result.data,
                columns: result.columns,
                rowCount: result.row_count,
                columnCount: result.column_count,
                displayName: result.display_name,
                loading: false,
                dataLoaded: true,
                error: null,
            }));
        },
        [],
    );

    // Load tables list and config on mount (but NOT data)
    useEffect(() => {
        if (state.tablesLoaded) return;
        const loadTablesAndConfig = async () => {
            // First load config
            const { config } = await fetchBrowseConfig();
            const batchSize = config?.browse_batch_size ?? DEFAULT_BATCH_SIZE;

            // Then load table list
            const { tables, error } = await fetchTableList();
            if (error) {
                setState(prev => ({ ...prev, tablesLoaded: true, limit: batchSize, error }));
                return;
            }
            const firstTable = tables.length > 0 ? tables[0].table_name : '';
            // Set tables but do NOT auto-fetch data (Rule 1: don't fetch on page load)
            setState(prev => ({
                ...prev,
                tables,
                selectedTable: firstTable,
                tablesLoaded: true,
                limit: batchSize,
                error: null
            }));
        };
        void loadTablesAndConfig();
    }, [state.tablesLoaded]);

    const onTableChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const tableName = event.target.value;
        // Rule 2: selecting a table always fetches data
        void loadTableData(tableName, state.limit, 0);
    };

    const refresh = () => {
        if (state.selectedTable) {
            void loadTableData(state.selectedTable, state.limit, state.offset);
        }
    };

    const goPrevious = () => {
        const nextOffset = Math.max(0, state.offset - state.limit);
        void loadTableData(state.selectedTable, state.limit, nextOffset);
    };

    const goNext = () => {
        const nextOffset = state.offset + state.limit;
        void loadTableData(state.selectedTable, state.limit, nextOffset);
    };

    return (
        <div className="dbb-page">
            {/* Header */}
            <div className="dbb-header">
                <div className="dbb-title">
                    <DatabaseIcon size={22} className="dbb-title-icon" />
                    <h1>Database Browser</h1>
                </div>
                <p className="dbb-subtitle">Browse tables stored in the application database.</p>
            </div>

            {/* Controls Row */}
            <div className="dbb-controls-row">
                <div className="dbb-controls-left">
                    <div className="dbb-control">
                        <label className="dbb-label">Select Table</label>
                        <div className="dbb-select-row">
                            <select
                                className="dbb-select"
                                value={state.selectedTable}
                                onChange={onTableChange}
                                disabled={state.loading || state.tables.length === 0}
                            >
                                {state.tables.map(table => (
                                    <option key={table.table_name} value={table.table_name}>
                                        {table.display_name}
                                    </option>
                                ))}
                            </select>
                            <button
                                className="dbb-refresh-btn"
                                onClick={refresh}
                                disabled={state.loading || !state.selectedTable}
                                title="Refresh data"
                            >
                                <RefreshCcw size={16} className={state.loading ? 'spinning' : ''} />
                            </button>
                        </div>
                    </div>
                </div>

                <div className="dbb-stats-row">
                    <span className="dbb-stats-label">Statistics</span>
                    <div className="dbb-stat-item">
                        <span className="dbb-stat-name">Rows:</span>
                        <span className="dbb-stat-value">{state.dataLoaded ? state.rowCount : '-'}</span>
                    </div>
                    <div className="dbb-stat-item">
                        <span className="dbb-stat-name">Columns:</span>
                        <span className="dbb-stat-value">{state.dataLoaded ? state.columnCount : '-'}</span>
                    </div>
                    <div className="dbb-stat-item">
                        <span className="dbb-stat-name">Table:</span>
                        <span className="dbb-stat-value">{state.dataLoaded ? (state.displayName || '-') : '-'}</span>
                    </div>
                </div>
            </div>

            {/* Error Message */}
            {state.error && <div className="dbb-error">{state.error}</div>}

            {/* Data Table */}
            <div className="dbb-table-wrapper">
                {state.loading ? (
                    <div className="dbb-loading">Loading...</div>
                ) : !state.dataLoaded ? (
                    <div className="dbb-empty">
                        Select a table and click refresh, or choose a different table to view data.
                    </div>
                ) : state.rows.length > 0 ? (
                    <div className="dbb-table-scroll">
                        <table className="dbb-table">
                            <thead>
                                <tr>
                                    <th className="dbb-row-num-header">#</th>
                                    {state.columns.map(col => (
                                        <th key={col}>{col}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {state.rows.map((row, idx) => (
                                    <tr key={idx}>
                                        <td className="dbb-row-num">{state.offset + idx + 1}</td>
                                        {state.columns.map(col => (
                                            <td key={col}>
                                                {row[col] !== null && row[col] !== undefined ? String(row[col]) : ''}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <div className="dbb-empty">
                        {state.tablesLoaded && state.tables.length === 0
                            ? 'No tables found in the connected database.'
                            : 'No data to display.'}
                    </div>
                )}
            </div>

            {/* Pagination */}
            {state.dataLoaded && state.rows.length > 0 && (
                <div className="dbb-pagination">
                    <button className="dbb-page-btn" onClick={goPrevious} disabled={!canGoPrevious}>
                        Previous
                    </button>
                    <span className="dbb-page-info">
                        Page {state.rowCount === 0 ? 0 : pageIndex + 1} · Showing {pageStart}-{pageEnd} of {state.rowCount}
                    </span>
                    <button className="dbb-page-btn" onClick={goNext} disabled={!canGoNext}>
                        Next
                    </button>
                </div>
            )}
        </div>
    );
}
